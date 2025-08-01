use candle_core::{display, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config as ConfigBase, ModelForCausalLM as ModelBase};
use tokenizers::Tokenizer;

use crate::{MAX_NEW_TOKENS, PROMPT};

pub fn candle(model_file: &str, config_file: &str, tokenizer_file: &str) {
    let dtype = DType::F32;
    let device = Device::Cpu;

    // Print everything, 8 decimal digits.
    display::set_print_options(display::PrinterOptions {
        precision: 10,
        threshold: usize::MAX, // don’t summarise
        edge_items: 3,
        line_width: 120,
        sci_mode: None,
    });

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device) }.unwrap();

    let config_data = std::fs::read(config_file).unwrap();
    let config: ConfigBase = serde_json::from_slice(&config_data).unwrap();

    let mut logits_processor = LogitsProcessor::new(1024, Some(0.67), Some(1.0));

    let mut model = ModelBase::new(&config, vb).unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_file).unwrap();
    let tokens = tokenizer.encode(PROMPT, true).unwrap().get_ids().to_vec();
    let mut tokens = tokens.iter().map(|x| *x as u32).collect::<Vec<_>>();
    let mut pos: usize = 0;
    for index in 0..MAX_NEW_TOKENS {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &device).unwrap().unsqueeze(0).unwrap();
        let logits = model.forward(&input, start_pos).unwrap();
        let logits = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let next_token = logits_processor.sample(&logits).unwrap();
        tokens.push(next_token);
    }

    let output_text = tokenizer.decode(&tokens, true).unwrap();
    println!("\nGenerated text:\n{}", output_text);
}
