//! Nexto/Necto's `EARLPerceiver` + `ControlsPredictorDot`, ported to candle.
//!
//! Structure read from the shipped TorchScript (not assumed):
//! ```text
//! q  = query_preprocess(q)        Linear(32->128), ReLU, Linear(128->128), ReLU
//! kv = key_value_preprocess(kv)   Linear(24->128), ReLU, Linear(128->128), ReLU
//! for block in blocks:            (2 blocks, both attending the SAME kv)
//!     q = q + attn(norm1(q), norm2(kv), key_padding_mask)
//!     q = q + linear2(relu(linear1(norm3(q))))
//! logits = einsum("ad,bpd->bpa", action_emb, emb_convertor(relu(q)))[:, 0, :]
//! ```
//! `action_emb` is `output.net` (Linear(8->32), ReLU, Linear(32->32), ReLU,
//! Linear(32->32), ReLU) applied to the 90-row action TABLE -- a constant, so it
//! is computed once at load time.
//!
//! Attention is 4-head over d_model 128 (head_dim 32) with Q,K,V packed into a
//! single `in_proj_weight` [384,128]; we split it and reuse `policy_v1::Mha`,
//! whose maths (scale, softmax, additive mask) already matches PyTorch's.
use candle_core::{Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module};
use std::collections::HashMap;

use crate::policy_v1::Mha;

pub const NEXTO_D_MODEL: usize = 128;
pub const NEXTO_HEADS: usize = 4;
pub const NEXTO_ACTIONS: usize = 90;
const LN_EPS: f64 = 1e-5;
/// Additive mask value for padded entities (matches policy_v1's convention).
const MASK_NEG: f32 = -1e9;

type Raw = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn e<T>(r: candle_core::Result<T>) -> Result<T, String> {
    r.map_err(|x| x.to_string())
}

fn tensor(map: &Raw, key: &str, dev: &Device) -> Result<Tensor, String> {
    let (data, shape) = map.get(key).ok_or_else(|| format!("missing key: {key}"))?;
    Tensor::from_vec(data.clone(), shape.clone(), dev).map_err(|x| format!("{key}: {x}"))
}

fn linear(map: &Raw, prefix: &str, dev: &Device) -> Result<Linear, String> {
    let w = tensor(map, &format!("{prefix}.weight"), dev)?;
    let b = tensor(map, &format!("{prefix}.bias"), dev)?;
    Ok(Linear::new(w, Some(b)))
}

fn layer_norm(map: &Raw, prefix: &str, dev: &Device) -> Result<LayerNorm, String> {
    let w = tensor(map, &format!("{prefix}.weight"), dev)?;
    let b = tensor(map, &format!("{prefix}.bias"), dev)?;
    Ok(LayerNorm::new(w, b, LN_EPS))
}

struct NexBlock {
    /// Nexto is pre-norm (norm1/2/3 present); Necto's block has NO LayerNorms at
    /// all -- plain residual attention + FF. Detected from the state dict.
    norm1: Option<LayerNorm>,
    norm2: Option<LayerNorm>,
    norm3: Option<LayerNorm>,
    attn: Mha,
    linear1: Linear,
    linear2: Linear,
}

impl NexBlock {
    fn new(map: &Raw, prefix: &str, dev: &Device) -> Result<Self, String> {
        // split the packed in_proj [3*d, d] into Q, K, V
        let w = tensor(map, &format!("{prefix}.attention.in_proj_weight"), dev)?;
        let b = tensor(map, &format!("{prefix}.attention.in_proj_bias"), dev)?;
        let d = NEXTO_D_MODEL;
        let mut parts = Vec::with_capacity(3);
        for i in 0..3 {
            let wi = e(w.narrow(0, i * d, d))?;
            let bi = e(b.narrow(0, i * d, d))?;
            parts.push(Linear::new(e(wi.contiguous())?, Some(e(bi.contiguous())?)));
        }
        let mut it = parts.into_iter();
        let (q, k, v) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        let o = linear(map, &format!("{prefix}.attention.out_proj"), dev)?;
        let opt_norm = |name: &str| -> Result<Option<LayerNorm>, String> {
            if map.contains_key(&format!("{prefix}.{name}.weight")) {
                Ok(Some(layer_norm(map, &format!("{prefix}.{name}"), dev)?))
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            norm1: opt_norm("norm1")?,
            norm2: opt_norm("norm2")?,
            norm3: opt_norm("norm3")?,
            attn: Mha::from_parts(q, k, v, o, NEXTO_HEADS, d / NEXTO_HEADS),
            linear1: linear(map, &format!("{prefix}.linear1"), dev)?,
            linear2: linear(map, &format!("{prefix}.linear2"), dev)?,
        })
    }

    fn forward(&self, q: &Tensor, kv: &Tensor, mask_add: &Tensor) -> Result<Tensor, String> {
        let apply = |n: &Option<LayerNorm>, t: &Tensor| -> Result<Tensor, String> {
            match n {
                Some(ln) => e(ln.forward(t)),
                None => Ok(t.clone()),
            }
        };
        let qn = apply(&self.norm1, q)?;
        let kn = apply(&self.norm2, kv)?;
        let a = self.attn.forward(&qn, &kn, mask_add)?;
        let q = e(q.add(&a))?;
        let h = e(self.linear1.forward(&apply(&self.norm3, &q)?))?;
        let h = e(h.relu())?;
        let h = e(self.linear2.forward(&h))?;
        e(q.add(&h))
    }
}

/// Nexto and Necto share the EARL trunk but differ at the head.
enum EarlHead {
    /// Nexto: ControlsPredictorDot -- logits = action_emb . emb_convertor(relu(q)).
    Dot { emb_convertor: Linear, action_emb: Tensor },
    /// Necto: a single Linear(128 -> 12) split into 5 heads of sizes [3,3,2,2,2].
    MultiDiscrete { linear: Linear, sizes: Vec<usize> },
}

/// What the head produced: either a flat logit vector over the action table, or
/// one logit vector per multi-discrete head.
pub enum HeadOut {
    Dot(Vec<f32>),
    MultiDiscrete(Vec<Vec<f32>>),
}

pub struct NextoNet {
    q_pre: (Linear, Linear),
    kv_pre: (Linear, Linear),
    blocks: Vec<NexBlock>,
    head: EarlHead,
}

impl NextoNet {
    /// `map` keys are the TorchScript parameter names (`net.earl.*`, `net.output.*`).
    /// `action_table` is the 90x8 lookup (identical to our own `make_lookup_table`).
    pub fn new(map: &Raw, action_table: &[[f32; 8]]) -> Result<Self, String> {
        crate::policy::ensure_single_thread_gemm();
        let dev = Device::Cpu;
        let n_blocks = (0..)
            .take_while(|i| map.contains_key(&format!("net.earl.blocks.{i}.linear1.weight")))
            .count();
        if n_blocks == 0 {
            return Err("no earl blocks found in state dict".into());
        }
        let blocks = (0..n_blocks)
            .map(|i| NexBlock::new(map, &format!("net.earl.blocks.{i}"), &dev))
            .collect::<Result<Vec<_>, _>>()?;

        // Head: Nexto has emb_convertor + net (dot product over the action table);
        // Necto has a single output.linear split into [3,3,2,2,2].
        let head = if map.contains_key("net.output.emb_convertor.weight") {
            // action embeddings: output.net = Linear,ReLU,Linear,ReLU,Linear,ReLU
            let flat: Vec<f32> = action_table.iter().flatten().copied().collect();
            let n = action_table.len();
            let mut a = e(Tensor::from_vec(flat, (n, 8), &dev))?;
            for i in [0usize, 2, 4] {
                let l = linear(map, &format!("net.output.net.{i}"), &dev)?;
                a = e(e(l.forward(&a))?.relu())?;
            }
            EarlHead::Dot {
                emb_convertor: linear(map, "net.output.emb_convertor", &dev)?,
                action_emb: a,
            }
        } else if map.contains_key("net.output.linear.weight") {
            EarlHead::MultiDiscrete {
                linear: linear(map, "net.output.linear", &dev)?,
                sizes: vec![3, 3, 2, 2, 2],
            }
        } else {
            return Err("state dict has neither a Dot nor a MultiDiscrete head".into());
        };

        Ok(Self {
            q_pre: (
                linear(map, "net.earl.query_preprocess.0", &dev)?,
                linear(map, "net.earl.query_preprocess.2", &dev)?,
            ),
            kv_pre: (
                linear(map, "net.earl.key_value_preprocess.0", &dev)?,
                linear(map, "net.earl.key_value_preprocess.2", &dev)?,
            ),
            blocks,
            head,
        })
    }

    /// Nexto-only convenience: flat logits over the action table.
    pub fn forward(&self, q: &[f32], kv: &[f32], mask: &[bool]) -> Result<Vec<f32>, String> {
        match self.forward_head(q, kv, mask)? {
            HeadOut::Dot(v) => Ok(v),
            HeadOut::MultiDiscrete(_) => {
                Err("this net has a multi-discrete head; use forward_head".into())
            }
        }
    }

    /// `q`: 32 floats. `kv`: `n_ent * 24`. `mask[i]` true = entity is padding.
    pub fn forward_head(&self, q: &[f32], kv: &[f32], mask: &[bool]) -> Result<HeadOut, String> {
        let dev = Device::Cpu;
        let n_ent = mask.len();
        let mut x = e(Tensor::from_vec(q.to_vec(), (1, 1, q.len()), &dev))?;
        let mut y = e(Tensor::from_vec(kv.to_vec(), (1, n_ent, kv.len() / n_ent), &dev))?;

        x = e(e(self.q_pre.0.forward(&x))?.relu())?;
        x = e(e(self.q_pre.1.forward(&x))?.relu())?;
        y = e(e(self.kv_pre.0.forward(&y))?.relu())?;
        y = e(e(self.kv_pre.1.forward(&y))?.relu())?;

        let madd: Vec<f32> = mask.iter().map(|&m| if m { MASK_NEG } else { 0.0 }).collect();
        let mask_add = e(Tensor::from_vec(madd, (1, 1, 1, n_ent), &dev))?;

        for b in &self.blocks {
            x = b.forward(&x, &y, &mask_add)?;
        }

        let xr = e(x.relu())?;
        match &self.head {
            EarlHead::Dot { emb_convertor, action_emb } => {
                // logits = action_emb [A,32] . emb_convertor(relu(x)) [1,1,32]
                let emb = e(emb_convertor.forward(&xr))?;
                let emb = e(emb.reshape((emb.dims()[2], 1)))?; // [32,1]
                let logits = e(action_emb.matmul(&emb))?; // [A,1]
                Ok(HeadOut::Dot(e(e(logits.flatten_all())?.to_vec1::<f32>())?))
            }
            EarlHead::MultiDiscrete { linear, sizes } => {
                let out = e(linear.forward(&xr))?; // [1,1,12]
                let flat = e(e(out.flatten_all())?.to_vec1::<f32>())?;
                let mut heads = Vec::with_capacity(sizes.len());
                let mut off = 0usize;
                for &sz in sizes {
                    heads.push(flat[off..off + sz].to_vec());
                    off += sz;
                }
                Ok(HeadOut::MultiDiscrete(heads))
            }
        }
    }
}

/// Decode Necto's multi-discrete heads to controls-8, matching
/// deploy/external/necto/agent.py: per-head argmax, then
/// throttle=a0-1, steer=a1-1, pitch=a0, yaw=a1*(1-a4), roll=a1*a4,
/// jump=a2, boost=a3, handbrake=a4.
pub fn decode_multi_discrete(heads: &[Vec<f32>]) -> [f32; 8] {
    let am = |v: &[f32]| v.iter().enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0;
    let a: Vec<i32> = heads.iter().map(|h| am(h) as i32).collect();
    let (a0, a1) = (a[0] - 1, a[1] - 1);
    let (a2, a3, a4) = (a[2], a[3], a[4]);
    [
        a0 as f32,
        a1 as f32,
        a0 as f32,
        (a1 * (1 - a4)) as f32,
        (a1 * a4) as f32,
        a2 as f32,
        a3 as f32,
        a4 as f32,
    ]
}
