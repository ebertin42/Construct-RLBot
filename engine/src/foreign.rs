//! Foreign (ported community-bot) opponent primitives: a candle MLP with LeakyReLU
//! plus (assembled in a later task) the obs + action-table + prev-action wrapper.
//! Kept separate from policy.rs (our v0 MLP) because the activation (LeakyReLU) and
//! the I/O contract differ. Weight-name contract: `net.{0,2,4,...}.{weight,bias}`
//! (torch Sequential of Linear + LeakyReLU), matching Immortal's jit.pt.
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use std::collections::HashMap;

const LEAKY_SLOPE: f64 = 0.01;

fn linear(
    w: &(Vec<f32>, Vec<usize>),
    b: &(Vec<f32>, Vec<usize>),
    dev: &Device,
) -> Result<Linear, String> {
    if w.1.len() != 2 {
        return Err(format!("weight must be 2-D, got {:?}", w.1));
    }
    let (out_dim, in_dim) = (w.1[0], w.1[1]);
    let wt = Tensor::from_vec(w.0.clone(), (out_dim, in_dim), dev).map_err(|e| e.to_string())?;
    let bt = Tensor::from_vec(b.0.clone(), out_dim, dev).map_err(|e| e.to_string())?;
    Ok(Linear::new(wt, Some(bt)))
}

pub struct ForeignMlp {
    layers: Vec<Linear>,
}

impl ForeignMlp {
    /// Build from a state_dict keyed `net.0.weight/bias, net.2..., net.{2*n_hidden}...`.
    /// Validates the chain dims: in_dim -> hidden (x n_hidden) -> out_dim.
    pub fn from_named(
        w: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        n_hidden: usize,
    ) -> Result<Self, String> {
        crate::policy::ensure_single_thread_gemm();
        let dev = Device::Cpu;
        let n_layers = n_hidden + 1;
        let mut layers = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let key = format!("net.{}", 2 * li);
            let wk = format!("{key}.weight");
            let bk = format!("{key}.bias");
            let wv = w.get(&wk).ok_or_else(|| format!("missing {wk}"))?;
            let bv = w.get(&bk).ok_or_else(|| format!("missing {bk}"))?;
            let expect_in = if li == 0 { in_dim } else { hidden };
            let expect_out = if li == n_layers - 1 { out_dim } else { hidden };
            if wv.1 != vec![expect_out, expect_in] {
                return Err(format!("{wk} shape {:?} != [{expect_out},{expect_in}]", wv.1));
            }
            layers.push(linear(wv, bv, &dev)?);
        }
        Ok(Self { layers })
    }

    /// Forward `batch` rows of `in_dim` -> flat `batch*out_dim` logits.
    /// LeakyReLU(0.01) after every layer except the last, via the identity
    /// `leaky(x) = relu(x) - alpha * relu(-x)`.
    pub fn forward(&self, obs: &[f32], batch: usize, in_dim: usize) -> Result<Vec<f32>, String> {
        let dev = Device::Cpu;
        let mut x =
            Tensor::from_vec(obs.to_vec(), (batch, in_dim), &dev).map_err(|e| e.to_string())?;
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(&x).map_err(|e| e.to_string())?;
            if i != last {
                let pos = x.relu().map_err(|e| e.to_string())?;
                let neg = x
                    .neg()
                    .and_then(|t| t.relu())
                    .and_then(|t| t.affine(LEAKY_SLOPE, 0.0))
                    .map_err(|e| e.to_string())?;
                x = (pos - neg).map_err(|e| e.to_string())?;
            }
        }
        x.flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| e.to_string())
    }
}
