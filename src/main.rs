#![recursion_limit = "256"]
use crate::candle::candle;
use crate::qwen2::{Qwen2Config, Qwen2Model};
use crate::sampling::{Sampler, TopP};
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{LibTorch, Metal, Wgpu};
use burn::tensor::Device;
use burn::{
    backend::candle::Candle,
    config::Config,
    module::Module,
    tensor::{activation, backend::Backend, Int, Shape, Tensor, TensorData},
};
use burn_ndarray::NdArray;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
mod bpe;
mod candle;
mod qwen2;
mod sampling;

const MAX_NEW_TOKENS: usize = 50;
const PROMPT: &str = "the quick brown";

fn main() {
    let api = Api::new().unwrap();
    let repo = api.model("Qwen/Qwen2-0.5B".to_string());
    let model_file = repo.get("model.safetensors").unwrap();
    let config_file = repo.get("config.json").unwrap();
    let tokenizer_file = repo.get("tokenizer.json").unwrap();

    // get the first argument
    let args: Vec<String> = std::env::args().collect();
    let lib = args.get(1).unwrap();

    if lib == "burn" {
        burn(
            model_file.to_str().unwrap(),
            config_file.to_str().unwrap(),
            tokenizer_file.to_str().unwrap(),
        );
    } else if lib == "candle" {
        candle(
            model_file.to_str().unwrap(),
            config_file.to_str().unwrap(),
            tokenizer_file.to_str().unwrap(),
        );
    }
}

fn burn(model_file: &str, config_file: &str, tokenizer_file: &str) {
    type MyBackend = Candle;
    let device = Device::<MyBackend>::default();

    println!(
        "config: {:?}",
        std::fs::read_to_string(&config_file).unwrap()
    );

    let config = Qwen2Config::load(config_file).unwrap();
    println!("config: {}", config);
    let tokenizer = Tokenizer::from_file(tokenizer_file).unwrap();

    let model: Qwen2Model<MyBackend> = Qwen2Model::new(&config, &device);
    let mut model = model.load(model_file).unwrap();

    let mut tokens = tokenizer.encode(PROMPT, true).unwrap().get_ids().to_vec();

    println!("Prompt token ids: {:?}", tokens);

    let mut sampler = Sampler::TopP(TopP::new(0.95, 1024));
    let mut pos: usize = 0;

    for _ in 0..MAX_NEW_TOKENS {
        let slice: Vec<u32> = tokens.iter().skip(pos).cloned().collect();
        let seq_len = slice.len();

        // Build the input tensor [1, seq_len]
        let shape = Shape::new([1, seq_len]);
        let x: Tensor<MyBackend, 2, Int> =
            Tensor::from_data(TensorData::new(slice, shape), &device);

        let output = model.forward(x, pos);
        pos += seq_len;

        let next_token_tensor = sampler.sample(output.squeeze(1), 0.67);
        let next_token_id: u32 = next_token_tensor.to_data().iter::<i64>().next().unwrap() as u32;
        tokens.push(next_token_id);
    }

    let output_text = tokenizer.decode(&tokens, true).unwrap();
    println!("\nGenerated text:\n{}", output_text);
}
