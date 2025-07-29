use burn::tensor::{activation, backend::Backend, Int, Tensor};
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};

pub enum Sampler<B: Backend> {
    TopP(TopP<B>),
    Argmax,
}

impl<B: Backend> Sampler<B> {
    pub fn sample(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        match self {
            Self::TopP(s) => {
                // Convert logits to probabilities using softmax
                let probs = activation::softmax(logits, 1);
                s.sample(probs)
            }
            Self::Argmax => logits.argmax(1),
        }
    }
}

pub trait Sampling<B: Backend> {
    fn sample(&mut self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int>;
}

/// Top-p sampling (nucleus sampling) selects the smallest set of tokens whose cumulative
/// probability mass exceed the threshold p.
pub struct TopP<B: Backend> {
    /// Probability threshold for sampling.
    p: f64,
    /// RNG.
    rng: StdRng,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> TopP<B> {
    pub fn new(p: f64, seed: u64) -> Self {
        let rng = StdRng::from_entropy();
        Self {
            p,
            rng,
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Sampling<B> for TopP<B> {
    fn sample(&mut self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        assert_eq!(
            probs.dims()[0],
            1,
            "Naive top-p sampling only supports single-batch tensors"
        );

        let (probs_sort, probs_idx) = probs.sort_descending_with_indices(1);

        // TODO: cumsum + Distribution::Multinomial support
        let mut probs_sort = probs_sort.to_data().iter::<f64>().collect::<Vec<_>>();
        let mut cumsum = 0.;
        probs_sort.iter_mut().for_each(|x| {
            if cumsum >= self.p {
                *x = 0.0;
            } else {
                cumsum += *x as f64;
            }
        });

        let dist = WeightedIndex::new(probs_sort).unwrap();
        let next_token_idx = dist.sample(&mut self.rng);

        probs_idx.slice([0..1, next_token_idx..next_token_idx + 1])
    }
}
