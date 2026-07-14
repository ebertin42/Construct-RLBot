use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use std::sync::Once;

static GEMM_GUARD: Once = Once::new();

/// candle's CPU matmul spins a process-wide rayon pool sized by
/// RAYON_NUM_THREADS (default: physical cores). With 14 worker threads each
/// doing their own small matmuls, that pool oversubscribes the box — force
/// single-threaded gemm unless the user overrode it. Must run before the
/// first tensor op (candle reads the var lazily into a OnceLock).
pub fn ensure_single_thread_gemm() {
    GEMM_GUARD.call_once(|| {
        if std::env::var("RAYON_NUM_THREADS").is_err() {
            std::env::set_var("RAYON_NUM_THREADS", "1");
        }
    });
}

pub struct LayerWeights {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

pub struct PolicyWeights {
    pub trunk: Vec<LayerWeights>,
    pub policy: LayerWeights,
    pub value: LayerWeights,
}

pub struct MlpPolicy {
    trunk: Vec<Linear>,
    policy_head: Linear,
    value_head: Linear,
}

fn linear(l: &LayerWeights, dev: &Device) -> Result<Linear, String> {
    if l.w.len() != l.out_dim * l.in_dim || l.b.len() != l.out_dim {
        return Err(format!(
            "layer shape mismatch: w={} b={} expected {}x{}",
            l.w.len(), l.b.len(), l.out_dim, l.in_dim
        ));
    }
    let w = Tensor::from_vec(l.w.clone(), (l.out_dim, l.in_dim), dev).map_err(|e| e.to_string())?;
    let b = Tensor::from_vec(l.b.clone(), l.out_dim, dev).map_err(|e| e.to_string())?;
    Ok(Linear::new(w, Some(b)))
}

impl MlpPolicy {
    pub fn new(weights: &PolicyWeights) -> Result<Self, String> {
        ensure_single_thread_gemm();
        let dev = Device::Cpu;
        Ok(Self {
            trunk: weights.trunk.iter().map(|l| linear(l, &dev)).collect::<Result<_, _>>()?,
            policy_head: linear(&weights.policy, &dev)?,
            value_head: linear(&weights.value, &dev)?,
        })
    }

    /// obs is a flat row-major batch*obs_dim buffer.
    /// Returns (logits flat batch*action_count, values batch).
    pub fn forward(
        &self,
        obs: &[f32],
        batch: usize,
        obs_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let dev = Device::Cpu;
        let mut x = Tensor::from_vec(obs.to_vec(), (batch, obs_dim), &dev)
            .map_err(|e| e.to_string())?;
        for l in &self.trunk {
            x = l.forward(&x).and_then(|t| t.relu()).map_err(|e| e.to_string())?;
        }
        let logits = self.policy_head.forward(&x).map_err(|e| e.to_string())?;
        let values = self.value_head.forward(&x).map_err(|e| e.to_string())?;
        let logits = logits.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?;
        let values = values.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?;
        Ok((logits, values))
    }
}
