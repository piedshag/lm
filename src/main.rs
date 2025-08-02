#![recursion_limit = "256"]
use crate::qwen2::{Qwen2Config, Qwen2Model};
use crate::sampling::{Sampler, TopP};
use burn::backend::{Metal, Wgpu};
use burn::tensor::Device;
use burn::{
    backend::candle::Candle,
    config::Config,
    module::Module,
    tensor::{activation, backend::Backend, Int, Shape, Tensor, TensorData},
};
use burn_ndarray::NdArray;
use clap::Parser;
use hf_hub::api::sync::Api;
use std::io::{self, Write};
use std::time::Instant;
use tokenizers::Tokenizer;
mod bpe;
mod qwen2;
mod sampling;
mod utils;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to generate text from
    #[arg(short, long, default_value = "what is the meaning of life?")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[arg(short, long, default_value_t = 100)]
    max_tokens: usize,
}

fn main() {
    let args = Args::parse();

    type MyBackend = Candle;
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
    let tokenizer = Tokenizer::from_file(tokenizer_file).unwrap();

    let model: Qwen2Model<MyBackend> = Qwen2Model::new(&config, &device);
    let model = model.load(&model_file).unwrap();

    print!("{}", args.prompt);
    io::stdout().flush().unwrap();

    let start = Instant::now();
    let mut tokens_generated = 0u32;
    for token_piece in generate(model, tokenizer, &args.prompt, args.max_tokens) {
        print!("{}", token_piece);
        io::stdout().flush().unwrap();
        tokens_generated += 1;
    }
    println!();

    let elapsed_secs = start.elapsed().as_secs_f64();
    let tps = tokens_generated as f64 / elapsed_secs;
    println!(
        "Generated {} tokens in {:.2} seconds ({:.2} tokens/s)",
        tokens_generated, elapsed_secs, tps
    );
}

pub struct GenerateStream<B: Backend> {
    model: Qwen2Model<B>,
    tokenizer: Tokenizer,
    sampler: Sampler<B>,
    tokens: Vec<u32>,
    pos: usize,
    remaining: usize,
}

impl<B: Backend> Iterator for GenerateStream<B> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let slice: Vec<u32> = self.tokens.iter().skip(self.pos).cloned().collect();
        let seq_len = slice.len();

        let x: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(slice, [1, seq_len]), &self.model.device);

        let output = self.model.forward(x, self.pos);
        self.pos += seq_len;

        let next_token_tensor = self.sampler.sample(output.squeeze(1), 0.67);
        let next_token_id: u32 = next_token_tensor.to_data().iter::<i64>().next().unwrap() as u32;
        self.tokens.push(next_token_id);
        self.remaining -= 1;

        let text_piece = self.tokenizer.decode(&[next_token_id], true).unwrap();
        Some(text_piece)
    }
}

fn generate<B: Backend>(
    model: Qwen2Model<B>,
    tokenizer: Tokenizer,
    prompt: &str,
    max_tokens: usize,
) -> GenerateStream<B> {
    let initial_tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();

    GenerateStream {
        model,
        tokenizer,
        sampler: Sampler::TopP(TopP::new(0.99, 1024)),
        tokens: initial_tokens,
        pos: 0,
        remaining: max_tokens,
    }
}
