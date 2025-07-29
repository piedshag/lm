use burn::{
    config::Config,
    module::{Module, Param},
    nn::{self, Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RotaryEncodingConfig},
    record::{FileRecorder, HalfPrecisionSettings, Recorder, RecorderError},
    serde::Deserialize,
    tensor::{activation, backend::Backend, Bool, Device, Int, Tensor},
};
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};

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
    #[config(default = "1e-6")]
    pub rms_norm_eps: f64,
}

#[derive(Debug)]
pub struct Qwen2Model<B: Backend> {
    pub model: Qwen2ModelBase<B>,
    pub lm_head: nn::Linear<B>,
    pub rotary_emb: nn::RotaryEncoding<B>,
    pub cache: Vec<KeyValueCache<B>>,
    pub device: Device<B>,
}

impl<B: Backend> Qwen2Model<B> {
    pub fn new(config: &Qwen2Config, device: &Device<B>) -> Self {
        let model = Qwen2ModelBase::new(config, device);
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_emb = RotaryEncodingConfig::new(config.max_position_embeddings, head_dim)
            .with_theta(config.rope_theta)
            .init(device);

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

    pub fn load(mut self, file_path: &str) -> Result<Self, RecorderError> {
        let recorder = SafetensorsFileRecorder::<HalfPrecisionSettings>::new();
        let load_args = LoadArgs::new(file_path.into())
            .with_key_remap("model\\.(.+)", "$1")
            .with_key_remap("(.*)norm\\.weight", "${1}norm.gamma");

        let record = recorder.load(load_args, &self.device).unwrap();
        self.model = self.model.load_record(record);

        // The small model uses tied embeddings, so we need to set the lm_head.weight to the
        // embed_tokens.weight
        let weights = self.model.embed_tokens.weight.val();
        self.lm_head.weight = Param::from_tensor(weights.transpose());

        Ok(self)
    }

    pub fn forward(&mut self, x: Tensor<B, 2, Int>, pos: usize) -> Tensor<B, 3> {
        let now = std::time::Instant::now();
        let hidden_states = self
            .model
            .forward(x, pos, &mut self.cache, &self.rotary_emb);
        let output = self.lm_head.forward(hidden_states);
        println!("forward time: {:?}", now.elapsed());
        output
    }
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
        rotary_emb: &nn::RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let [_batch_size, seq_len] = x.dims();
        let mut x = self.embed_tokens.forward(x);

        let mask = if seq_len > 1 {
            let mask: Tensor<B, 2, Int> = Tensor::ones([seq_len, seq_len], &x.device()).triu(1);
            Some(mask.equal_elem(1))
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, pos, &mut cache[i], mask.as_ref(), rotary_emb);
        }

        self.norm.forward(x)
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
        mask: Option<&Tensor<B, 2, Bool>>,
        rotary_emb: &nn::RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let x_norm = self.input_layernorm.forward(x);
        let attn_output = self.self_attn.forward(x_norm, pos, cache, mask, rotary_emb);
        let x = residual + attn_output;

        let residual = x.clone();
        let x_norm = self.post_attention_layernorm.forward(x);
        let mlp_output = self.mlp.forward(x_norm);
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
        mask: Option<&Tensor<B, 2, Bool>>,
        rotary_emb: &nn::RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden_size] = x.dims();

        let query_states = self.q_proj.forward(x.clone());
        let key_states = self.k_proj.forward(x.clone());
        let value_states = self.v_proj.forward(x);

        let query_states = query_states
            .reshape([batch_size, seq_len, self.num_attention_heads, self.head_dim])
            .swap_dims(1, 2); // [batch_size, num_heads, seq_len, head_dim]
        let key_states = key_states
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2); // [batch_size, num_kv_heads, seq_len, head_dim]
        let value_states = value_states
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2); // [batch_size, num_kv_heads, seq_len, head_dim]

        let query_states = rotary_emb.apply::<4>(query_states, pos);
        let key_states = rotary_emb.apply::<4>(key_states, pos);

        let (key_states, value_states) = cache.update(key_states, value_states);
        let attn_output = self.attention(query_states, key_states, value_states, mask);

        let attn_output = attn_output.swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ]);
        self.o_proj.forward(attn_output)
    }

    fn attention(
        &self,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        mask: Option<&Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 4> {
        let [_batch_size, num_heads, _q_len, head_dim] = query.dims();

        let key = self.repeat_kv(key, num_heads / self.num_key_value_heads);
        let value = self.repeat_kv(value, num_heads / self.num_key_value_heads);

        let mut attn_weights = query.matmul(key.swap_dims(2, 3)) / (head_dim as f64).sqrt();

        if let Some(mask) = mask {
            attn_weights = attn_weights.mask_fill(
                mask.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0),
                -f32::INFINITY,
            );
        }

        let attn_weights = activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(value);
        attn_output
    }

    fn repeat_kv(&self, x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        if n_rep == 1 {
            return x;
        }
        let [batch_size, n_kv_heads, seq_len, head_dim] = x.dims();
        x.unsqueeze_dim::<5>(2)
            .expand([batch_size, n_kv_heads, n_rep, seq_len, head_dim])
            .reshape([batch_size, n_kv_heads * n_rep, seq_len, head_dim])
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
