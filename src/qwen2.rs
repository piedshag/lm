use std::path::Path;

use crate::utils::print_tensor;
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self, attention::generate_autoregressive_mask, Embedding, EmbeddingConfig, Linear,
        LinearConfig, RmsNorm, RotaryEncoding,
    },
    record::{FileRecorder, FullPrecisionSettings, HalfPrecisionSettings, Recorder, RecorderError},
    serde::Deserialize,
    tensor::{
        activation, backend::Backend, Bool, Device, Int, Tensor, TensorData, TensorPrimitive,
    },
};
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};
use burn_ndarray::NdArrayTensor;
use rand::seq;

fn rotate_half<B: Backend>(xs: Tensor<B, 4>) -> Tensor<B, 4> {
    let last_dim = xs.dims()[3];
    let xs1 = xs.clone().narrow(3, 0, last_dim / 2);
    let xs2 = xs.narrow(3, last_dim / 2, last_dim - last_dim / 2);
    Tensor::cat(vec![xs2.neg(), xs1], 3)
}

pub fn rope_slow<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [_b_sz, _h, seq_len, _n_embd] = x.dims();
    let cos = Tensor::cat(vec![cos.clone(), cos], 1);
    let sin = Tensor::cat(vec![sin.clone(), sin], 1);
    let cos = cos.narrow(0, 0, seq_len);
    let sin = sin.narrow(0, 0, seq_len);
    let cos = cos.unsqueeze();
    let sin = sin.unsqueeze();
    x.clone().mul(cos) + rotate_half(x).mul(sin)
}

#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        device: &Device<B>,
    ) -> Self {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_data(TensorData::new(inv_freq, [1, inv_freq_len]), device);
        let t: Tensor<B, 2> = Tensor::arange(0i64..max_position_embeddings as i64, device)
            .float()
            .reshape([max_position_embeddings, 1]);

        let freqs = t.matmul(inv_freq);
        let sin = freqs.clone().sin();
        let cos = freqs.cos();

        Self { sin, cos }
    }

    pub fn forward(&self, x: Tensor<B, 4>, offset: usize) -> Tensor<B, 4> {
        let [_b_sz, _qh, seq_len, _n_embd] = x.dims();
        let cos = self.cos.clone().narrow(0, offset, seq_len);
        let sin = self.sin.clone().narrow(0, offset, seq_len);

        rope_slow(x, cos, sin)
    }
}

/// Cache for key-value pairs in attention layers.
#[derive(Debug, Clone)]
pub struct KeyValueCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KeyValueCache<B> {
    /// Create a new, empty cache.
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Update the cache with new key-value pairs.
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (key_states, value_states) = match (&self.k, &self.v) {
            (None, None) => (k.clone(), v.clone()),
            (Some(prev_k), Some(prev_v)) => {
                let key_states = Tensor::cat(vec![prev_k.clone(), k], 2);
                let value_states = Tensor::cat(vec![prev_v.clone(), v], 2);
                (key_states, value_states)
            }
            _ => panic!("Inconsistent cache state"),
        };

        self.k = Some(key_states.clone());
        self.v = Some(value_states.clone());
        (key_states, value_states)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }
}

#[derive(Config)]
pub struct Qwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
}

#[derive(Debug)]
pub struct Qwen2Model<B: Backend> {
    pub model: Qwen2ModelBase<B>,
    pub lm_head: nn::Linear<B>,
    pub rotary_emb: RotaryEmbedding<B>,
    pub cache: Vec<KeyValueCache<B>>,
    pub device: Device<B>,
}

impl<B: Backend> Qwen2Model<B> {
    pub fn new(config: &Qwen2Config, device: &Device<B>) -> Self {
        let model = Qwen2ModelBase::new(config, device);
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_emb = RotaryEmbedding::new(
            config.rope_theta,
            head_dim,
            config.max_position_embeddings,
            device,
        );

        let lm_head = nn::LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        let cache = (0..config.num_hidden_layers)
            .map(|_| KeyValueCache::<B>::new())
            .collect::<Vec<_>>();

        Self {
            model,
            lm_head,
            rotary_emb,
            cache,
            device: device.clone(),
        }
    }

    pub fn load(mut self, file_path: &Path) -> Result<Self, RecorderError> {
        let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::new();
        let load_args = LoadArgs::new(file_path.into())
            .with_key_remap("model\\.(.+)", "$1")
            .with_key_remap("(.*)norm\\.weight", "${1}norm.gamma")
            .with_adapter_type(AdapterType::PyTorch);

        let record = recorder.load(load_args, &self.device).unwrap();
        self.model = self.model.load_record(record);

        load_lm_head(&mut self)?;
        Ok(self)
    }

    pub fn forward(&mut self, x: Tensor<B, 2, Int>, pos: usize) -> Tensor<B, 3> {
        let seq_len = x.dims()[1];
        let hidden_states = self
            .model
            .forward(x, pos, &mut self.cache, &self.rotary_emb)
            .narrow(1, seq_len - 1, 1);

        print_tensor("hidden_states", &hidden_states);
        let output = self.lm_head.forward(hidden_states);
        print_tensor("output", &output);
        output
    }
}

pub fn load_lm_head<B: Backend>(model: &mut Qwen2Model<B>) -> Result<(), RecorderError> {
    // The small model uses tied embeddings, so we need to set the lm_head.weight to the
    // embed_tokens.weight

    // VERY IMPORTANT that you transpose the weights before creating the tensor
    // transpose creates a view of the tensor which needs to be reshaped when we forward pass
    // this ends up being very slow
    let weights = model
        .model
        .embed_tokens
        .weight
        .val()
        .transpose()
        .into_data();
    let tensor = Tensor::from_data(weights, &model.device);
    model.lm_head.weight = Param::from_tensor(tensor);

    Ok(())
}

#[derive(Module, Debug)]
pub struct Qwen2ModelBase<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

impl<B: Backend> Qwen2ModelBase<B> {
    pub fn new(config: &Qwen2Config, device: &Device<B>) -> Self {
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let layers = (0..config.num_hidden_layers)
            .map(|_| DecoderLayer::new(config, device))
            .collect();

        let norm = nn::RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        Self {
            embed_tokens,
            layers,
            norm,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        pos: usize,
        cache: &mut [KeyValueCache<B>],
        rotary_emb: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = x.dims();
        let mut x = self.embed_tokens.forward(x);

        // Log after embedding.
        print_tensor("[Model] after embed_tokens", &x);

        let mask = if seq_len > 1 {
            Some(generate_autoregressive_mask(
                batch_size,
                seq_len,
                &x.device(),
            ))
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            print_tensor(&format!("[Model] ↓ Layer {i:02}  input"), &x);
            x = layer.forward(x, pos, &mut cache[i], mask.as_ref(), rotary_emb);
            print_tensor(&format!("[Model] ↑ Layer {i:02}  output"), &x);
        }

        let x = self.norm.forward(x);

        print_tensor("[Model] Norm output", &x);

        x
    }
}

#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    self_attn: Attention<B>,
    mlp: Mlp<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> DecoderLayer<B> {
    pub fn new(config: &Qwen2Config, device: &Device<B>) -> Self {
        let self_attn = Attention::new(config, device);
        let mlp = Mlp::new(config, device);

        let input_layernorm = nn::RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        let post_attention_layernorm = nn::RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        pos: usize,
        cache: &mut KeyValueCache<B>,
        mask: Option<&Tensor<B, 3, Bool>>,
        rotary_emb: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        // Input tensor.
        let residual = x.clone();
        print_tensor("  [DecoderLayer]     input", &residual);

        let x_norm = self.input_layernorm.forward(x);
        print_tensor("  [DecoderLayer] after layernorm", &x_norm);

        let attn_output = self.self_attn.forward(x_norm, pos, cache, mask, rotary_emb);
        print_tensor("  [DecoderLayer] after self_attn", &attn_output);

        let x = attn_output + residual;

        print_tensor("  [DecoderLayer] post_attention_layernorm input", &x);

        let residual = x.clone();
        let x_norm = self.post_attention_layernorm.forward(x);

        print_tensor("  [DecoderLayer] post_attention_layernorm", &x_norm);

        let mlp_output = self.mlp.forward(x_norm);

        print_tensor("  [DecoderLayer] after MLP", &mlp_output);

        residual + mlp_output
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    o_proj: nn::Linear<B>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(config: &Qwen2Config, device: &Device<B>) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let q_proj =
            nn::LinearConfig::new(config.hidden_size, config.num_attention_heads * head_dim)
                .with_bias(true)
                .init(device);
        let k_proj =
            nn::LinearConfig::new(config.hidden_size, config.num_key_value_heads * head_dim)
                .with_bias(true)
                .init(device);
        let v_proj =
            nn::LinearConfig::new(config.hidden_size, config.num_key_value_heads * head_dim)
                .with_bias(true)
                .init(device);
        let o_proj =
            nn::LinearConfig::new(config.num_attention_heads * head_dim, config.hidden_size)
                .with_bias(false)
                .init(device);

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        pos: usize,
        cache: &mut KeyValueCache<B>,
        mask: Option<&Tensor<B, 3, Bool>>,
        rotary_emb: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden_size] = x.dims();

        let query_states = self.q_proj.forward(x.clone());
        // Debug: hash of raw Q projections.
        print_tensor("    [Attention] q_proj", &query_states);
        let key_states = self.k_proj.forward(x.clone());
        print_tensor("    [Attention] k_proj", &key_states);
        let value_states = self.v_proj.forward(x);
        print_tensor("    [Attention] v_proj", &value_states);

        let query_states = query_states
            .reshape([batch_size, seq_len, self.num_attention_heads, self.head_dim])
            .swap_dims(1, 2) // [batch_size, num_heads, seq_len, head_dim]
            .clone();
        let key_states = key_states
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2) // [batch_size, num_kv_heads, seq_len, head_dim]
            .clone();
        let value_states = value_states
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2) // [batch_size, num_kv_heads, seq_len, head_dim]
            .clone();

        print_tensor("    [Attention] Q reshaped", &query_states);
        print_tensor("    [Attention] K reshaped", &key_states);
        print_tensor("    [Attention] V reshaped", &value_states);

        let query_states = rotary_emb.forward(query_states, pos);
        let key_states = rotary_emb.forward(key_states, pos);

        print_tensor("    [Attention] after RoPE Q", &query_states);
        print_tensor("    [Attention] after RoPE K", &key_states);

        let (key_states, value_states) = cache.update(key_states, value_states);

        // Hashes after cache update.
        print_tensor("    [Attention] after cache K", &key_states);
        print_tensor("    [Attention] after cache V", &value_states);

        let attn_output = self.attention(query_states, key_states, value_states, mask);
        let attn_output = attn_output.swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ]);

        print_tensor("    [Attention] attn_output_reshape", &attn_output);

        self.o_proj.forward(attn_output)
    }

    fn attention(
        &self,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        mask: Option<&Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 4> {
        let [_batch_size, num_heads, _q_len, head_dim] = query.dims();

        let key = self.repeat_kv(key, num_heads / self.num_key_value_heads);
        let value = self.repeat_kv(value, num_heads / self.num_key_value_heads);

        // Use the exact numerical order Candle uses: multiply by the scale rather than divide.
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut attn_weights = query.matmul(key.swap_dims(2, 3)) * scale;

        if let Some(mask) = mask {
            let [_batch_size, num_heads, q_len, k_len] = attn_weights.dims();

            let expanded_mask = mask
                .clone()
                .unsqueeze_dim::<4>(0) // [1, 1, q_len, k_len]
                .expand([_batch_size, num_heads, q_len, k_len]);

            attn_weights = attn_weights.mask_fill(expanded_mask, f32::NEG_INFINITY);
        }

        let attn_weights = activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(value);

        print_tensor("attn_output", &attn_output);

        attn_output
    }

    fn repeat_kv(&self, x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        if n_rep == 1 {
            return x;
        }

        let [batch_size, n_kv_heads, seq_len, head_dim] = x.dims();
        let x_cat = Tensor::cat(vec![x; n_rep], 2);
        x_cat.reshape([batch_size, n_kv_heads * n_rep, seq_len, head_dim])
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    gate_proj: nn::Linear<B>,
    up_proj: nn::Linear<B>,
    down_proj: nn::Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &Qwen2Config, device: &Device<B>) -> Self {
        let gate_proj = nn::LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(false)
            .init(device);
        let up_proj = nn::LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(false)
            .init(device);
        let down_proj = nn::LinearConfig::new(config.intermediate_size, config.hidden_size)
            .with_bias(false)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = activation::silu(self.gate_proj.forward(x.clone())) * self.up_proj.forward(x);
        self.down_proj.forward(x)
    }
}
