#![recursion_limit = "256"]
use burn::{
    backend::ndarray::NdArray,
    config::Config,
    module::Module,
    tensor::{activation, backend::Backend, Device, Int, Shape, Tensor, TensorData},
};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::qwen2::{Qwen2Config, Qwen2Model};
use crate::sampling::{Sampler, TopP};

mod bpe;
mod qwen2;
mod sampling;

fn main() {
    type MyBackend = NdArray;
    let device = Device::<MyBackend>::default();

    let api = Api::new().unwrap();
    let repo = api.model("Qwen/Qwen2-0.5B".to_string());
    let model_file = repo.get("model.safetensors").unwrap();
    let config_file = repo.get("config.json").unwrap();
    let tokenizer_file = repo.get("tokenizer.json").unwrap();

    println!(
        "config: {:?}",
        std::fs::read_to_string(&config_file).unwrap()
    );

    let config = Qwen2Config::load(config_file).unwrap();
    println!("config: {}", config);
    let tokenizer = Tokenizer::from_file(tokenizer_file).unwrap();

    let model: Qwen2Model<MyBackend> = Qwen2Model::new(&config, &device);
    let mut model = model.load(model_file.to_str().unwrap()).unwrap();

    let prompt = "the quick brown";
    let mut tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();

    println!("Prompt token ids: {:?}", tokens);

    let mut sampler = Sampler::TopP(TopP::new(0.9, 20));
    let mut pos: usize = 0;

    let max_new_tokens = 1;
    for _ in 0..=max_new_tokens {
        let slice: Vec<u32> = tokens.iter().skip(pos).cloned().collect();
        let seq_len = slice.len();

        // Build the input tensor [1, seq_len]
        let shape = Shape::new([1, seq_len]);
        let x: Tensor<MyBackend, 2, Int> =
            Tensor::from_data(TensorData::new(slice, shape), &device);

        let output = model.forward(x, pos);
        pos += seq_len;

        let logits = output.slice([0..1, (seq_len - 1)..seq_len]);
        let next_token_tensor = sampler.sample(logits.squeeze(1));
        let next_token_id: u32 = next_token_tensor.to_data().iter::<i64>().next().unwrap() as u32;
        tokens.push(next_token_id);
    }

    let output_text = tokenizer.decode(&tokens, true).unwrap();
    println!("\nGenerated text:\n{}", output_text);
}
