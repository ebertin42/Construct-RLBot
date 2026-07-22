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
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
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
        Ok(Self {
            norm1: layer_norm(map, &format!("{prefix}.norm1"), dev)?,
            norm2: layer_norm(map, &format!("{prefix}.norm2"), dev)?,
            norm3: layer_norm(map, &format!("{prefix}.norm3"), dev)?,
            attn: Mha::from_parts(q, k, v, o, NEXTO_HEADS, d / NEXTO_HEADS),
            linear1: linear(map, &format!("{prefix}.linear1"), dev)?,
            linear2: linear(map, &format!("{prefix}.linear2"), dev)?,
        })
    }

    fn forward(&self, q: &Tensor, kv: &Tensor, mask_add: &Tensor) -> Result<Tensor, String> {
        let qn = e(self.norm1.forward(q))?;
        let kn = e(self.norm2.forward(kv))?;
        let a = self.attn.forward(&qn, &kn, mask_add)?;
        let q = e(q.add(&a))?;
        let h = e(self.linear1.forward(&e(self.norm3.forward(&q))?))?;
        let h = e(h.relu())?;
        let h = e(self.linear2.forward(&h))?;
        e(q.add(&h))
    }
}

pub struct NextoNet {
    q_pre: (Linear, Linear),
    kv_pre: (Linear, Linear),
    blocks: Vec<NexBlock>,
    emb_convertor: Linear,
    /// `output.net(action_table)` -> [90, 32]; the table is a constant so this is
    /// computed once.
    action_emb: Tensor,
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

        // action embeddings: output.net = Linear,ReLU,Linear,ReLU,Linear,ReLU
        let flat: Vec<f32> = action_table.iter().flatten().copied().collect();
        let n = action_table.len();
        let mut a = e(Tensor::from_vec(flat, (n, 8), &dev))?;
        for i in [0usize, 2, 4] {
            let l = linear(map, &format!("net.output.net.{i}"), &dev)?;
            a = e(e(l.forward(&a))?.relu())?;
        }

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
            emb_convertor: linear(map, "net.output.emb_convertor", &dev)?,
            action_emb: a,
        })
    }

    /// `q`: 32 floats. `kv`: `n_ent * 24`. `mask[i]` true = entity is padding.
    /// Returns `NEXTO_ACTIONS` logits.
    pub fn forward(&self, q: &[f32], kv: &[f32], mask: &[bool]) -> Result<Vec<f32>, String> {
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

        // logits = action_emb [A,32] . emb_convertor(relu(x)) [1,1,32]
        let emb = e(self.emb_convertor.forward(&e(x.relu())?))?; // [1,1,32]
        let emb = e(emb.reshape((emb.dims()[2], 1)))?; // [32,1]
        let logits = e(self.action_emb.matmul(&emb))?; // [A,1]
        e(e(logits.flatten_all())?.to_vec1::<f32>())
    }
}
