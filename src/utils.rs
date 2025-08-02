use burn::{prelude::Backend, tensor::Tensor};

pub fn print_tensor<B: Backend, const D: usize>(name: &str, tensor: &Tensor<B, D>) {
    return;
    println!(
        "{} shape {:?} hash {}",
        name,
        tensor.dims(),
        tensor_hash(tensor)
    );
}

fn tensor_hash<B: Backend, const D: usize>(t: &Tensor<B, D>) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let data = t.to_data();
    let mut hasher = DefaultHasher::new();
    for v in data.iter::<f32>() {
        let quantised = (v * 10_000.0).round() as i32;
        quantised.hash(&mut hasher);
    }

    format!("{:016x}", hasher.finish())
}
