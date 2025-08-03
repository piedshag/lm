# Rust Language Model

A simple Rust implementation of a language model using the Burn framework.

## How to Run

Build and run with default settings:
```bash
cargo run --release
```

This will generate text from the prompt "what is the meaning of life?" and produce up to 100 tokens.

### Feature Flags

The project supports different backends via feature flags:

- **CPU backend** (default): `cargo run --release --features cpu`
- **Metal backend** (macOS): `cargo run --release --features metal`

The Metal backend provides better performance on Apple Silicon Macs.

### Custom Options

Use a different prompt:
```bash
cargo run --features cpu --release -- --prompt "Write a short story"
```

Generate more tokens:
```bash
cargo run --features cpu --release -- --max-tokens 200
```

## Options

- `--prompt` or `-p`: The text prompt to generate from (default: "what is the meaning of life?")
- `--max-tokens` or `-m`: Maximum number of tokens to generate (default: 100)
- `--model` or `-m`: The model to use (default: qwen2-0-5b)
  - `qwen2-0-5b`: 0.5 billion parameter model
  - `qwen2-1-5b`: 1.5 billion parameter model 