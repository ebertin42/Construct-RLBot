//! Task T4 (entity-transformer plan, `docs/superpowers/plans/
//! 2026-07-16-entity-transformer-obs-v1.md`): candle inference for the obs-v1
//! entity-transformer policy/value net. This is a candle port of
//! `python/construct/learn/model_v1.py::EntityPolicyNet` -- it must reproduce
//! that net's `forward` NUMERICALLY (parity enforced by the golden fixture
//! test, `engine/tests/policy_v1_test.rs` + `scripts/gen_policy_v1_fixture.py`).
//!
//! ## Weight-name mapping (state_dict key -> this module)
//! Names are an EXACT, HARD contract with `model_v1.py`'s `state_dict()` --
//! do not rename either side without updating both.
//!
//! | state_dict key(s)                                   | field                  |
//! |-------------------------------------------------------|------------------------|
//! | `embed.weight` / `embed.bias`                          | `embed` (26 -> d)      |
//! | `query_embed.weight` / `.bias`                          | `query_embed` (64 -> d)|
//! | `act_embed.0.{weight,bias}`                             | `act_embed0` (8 -> d)  |
//! | `act_embed.2.{weight,bias}`                             | `act_embed2` (d -> 32) |
//! | `prev_embed_w`                                          | `prev_embed_w` ([5])   |
//! | `prev_proj.weight` / `.bias`                            | `prev_proj` (32 -> d)  |
//! | `blocks.{i}.ln1.{weight,bias}`                          | `blocks[i].ln1`        |
//! | `blocks.{i}.attn.{q,k,v,o}.{weight,bias}`               | `blocks[i].attn`       |
//! | `blocks.{i}.ln2.{weight,bias}`                          | `blocks[i].ln2`        |
//! | `blocks.{i}.ff1.{weight,bias}` / `ff2.{weight,bias}`    | `blocks[i].ff1/ff2`    |
//! | `pool_ln.{weight,bias}`                                 | `pool_ln`              |
//! | `pool.{q,k,v,o}.{weight,bias}`                          | `pool`                 |
//! | `policy_dot.weight` / `.bias`                           | `policy_dot` (d -> 32) |
//! | `value_head.weight` / `.bias`                           | `value_head` (d -> 1)  |
//! | `action_table`                                          | `action_table` ([N,8]) non-trainable buffer, consumed not errored on |
//!
//! `d_model`/`layers`/`ff`/table size `N` are all inferred from tensor
//! shapes; `heads` is NOT recoverable from shapes (attention splits d evenly
//! regardless of head count) so it is a required `EntityPolicy::new` param.
//!
//! aux heads (`aux_reward`/`aux_recon`) are config-gated OFF at launch
//! (per the plan) and are not loaded here; a state_dict containing them is
//! not currently supported (extra keys not in the mapping table above -- see
//! `action_table`, `dims` etc. exclusion list -- will surface as `Err` from a
//! downstream missing-tensor lookup, not silently ignored `aux_*` weights).

use candle_core::{Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module};
use std::collections::HashMap;

use crate::policy::ensure_single_thread_gemm;

const ACT_FEAT: usize = 8;
const ACT_EMB: usize = 32;
const LN_EPS: f64 = 1e-5;
/// Mirrors model_v1.py's MHA: `scores.masked_fill(key_mask, -1e9)` pre-softmax.
/// Implemented as an additive mask (`scores + (-1e9 or 0)`) rather than a
/// literal replace -- after softmax the two are numerically indistinguishable
/// in f32 (both underflow `exp()` to exactly 0.0 for any plausible score
/// magnitude), see module doc / T4 report for the reasoning.
const MASK_NEG: f32 = -1e9;

type RawTensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn tensor(map: &RawTensors, key: &str, dev: &Device) -> Result<Tensor, String> {
    let (data, shape) = map.get(key).ok_or_else(|| format!("missing state_dict key: {key}"))?;
    Tensor::from_vec(data.clone(), shape.clone(), dev).map_err(|e| format!("{key}: {e}"))
}

fn linear(map: &RawTensors, prefix: &str, dev: &Device) -> Result<Linear, String> {
    let w = tensor(map, &format!("{prefix}.weight"), dev)?;
    let b = tensor(map, &format!("{prefix}.bias"), dev)?;
    if w.dims().len() != 2 {
        return Err(format!("{prefix}.weight must be 2D, got {:?}", w.dims()));
    }
    Ok(Linear::new(w, Some(b)))
}

fn layer_norm(map: &RawTensors, prefix: &str, dev: &Device) -> Result<LayerNorm, String> {
    let w = tensor(map, &format!("{prefix}.weight"), dev)?;
    let b = tensor(map, &format!("{prefix}.bias"), dev)?;
    Ok(LayerNorm::new(w, b, LN_EPS))
}

/// Multi-head attention from primitives, matching `model_v1.py::MHA` exactly:
/// `q_in [B,Tq,d]`, `kv_in [B,Tk,d]`, additive key mask `[B,1,1,Tk]` (0 or
/// `MASK_NEG`) broadcast over heads and query positions.
struct Mha {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl Mha {
    fn new(map: &RawTensors, prefix: &str, d_model: usize, heads: usize, dev: &Device) -> Result<Self, String> {
        if d_model % heads != 0 {
            return Err(format!("d_model={d_model} not divisible by heads={heads}"));
        }
        Ok(Self {
            q: linear(map, &format!("{prefix}.q"), dev)?,
            k: linear(map, &format!("{prefix}.k"), dev)?,
            v: linear(map, &format!("{prefix}.v"), dev)?,
            o: linear(map, &format!("{prefix}.o"), dev)?,
            heads,
            head_dim: d_model / heads,
        })
    }

    fn forward(&self, q_in: &Tensor, kv_in: &Tensor, mask_add: &Tensor) -> Result<Tensor, String> {
        let qd = q_in.dims();
        let kd = kv_in.dims();
        let (b, tq) = (qd[0], qd[1]);
        let tk = kd[1];
        let e = |r: candle_core::Result<Tensor>| r.map_err(|e| e.to_string());

        let q = e(self.q.forward(q_in))?;
        let k = e(self.k.forward(kv_in))?;
        let v = e(self.v.forward(kv_in))?;

        let q = e(e(q.reshape((b, tq, self.heads, self.head_dim)))?.transpose(1, 2))?;
        let q = e(q.contiguous())?;
        let k = e(e(k.reshape((b, tk, self.heads, self.head_dim)))?.transpose(1, 2))?;
        let k = e(k.contiguous())?;
        let v = e(e(v.reshape((b, tk, self.heads, self.head_dim)))?.transpose(1, 2))?;
        let v = e(v.contiguous())?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let kt = e(k.transpose(2, 3))?;
        let kt = e(kt.contiguous())?;
        let scores = e(q.matmul(&kt))?;
        let scores = e(scores.affine(scale, 0.0))?; // [B,H,Tq,Tk]
        let scores = e(scores.broadcast_add(mask_add))?; // mask_add: [B,1,1,Tk]
        let attn = e(candle_nn::ops::softmax_last_dim(&scores))?;
        let out = e(attn.matmul(&v))?; // [B,H,Tq,hd]
        let out = e(out.transpose(1, 2))?;
        let out = e(out.contiguous())?;
        let out = e(out.reshape((b, tq, self.heads * self.head_dim)))?;
        e(self.o.forward(&out))
    }
}

/// Pre-LN transformer block: `x = x + attn(ln1(x)); x = x + ff2(relu(ff1(ln2(x))))`.
struct Block {
    ln1: LayerNorm,
    attn: Mha,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

impl Block {
    fn new(map: &RawTensors, prefix: &str, d_model: usize, heads: usize, dev: &Device) -> Result<Self, String> {
        Ok(Self {
            ln1: layer_norm(map, &format!("{prefix}.ln1"), dev)?,
            attn: Mha::new(map, &format!("{prefix}.attn"), d_model, heads, dev)?,
            ln2: layer_norm(map, &format!("{prefix}.ln2"), dev)?,
            ff1: linear(map, &format!("{prefix}.ff1"), dev)?,
            ff2: linear(map, &format!("{prefix}.ff2"), dev)?,
        })
    }

    fn forward(&self, x: &Tensor, mask_add: &Tensor) -> Result<Tensor, String> {
        let e = |r: candle_core::Result<Tensor>| r.map_err(|e| e.to_string());
        let h = e(self.ln1.forward(x))?;
        let a = self.attn.forward(&h, &h, mask_add)?;
        let x = e(x + a)?;
        let h2 = e(self.ln2.forward(&x))?;
        let f = e(self.ff1.forward(&h2))?;
        let f = e(f.relu())?;
        let f = e(self.ff2.forward(&f))?;
        e(&x + f)
    }
}

/// Candle inference for `model_v1.py::EntityPolicyNet`. See module doc for
/// the full weight-name mapping table.
pub struct EntityPolicy {
    embed: Linear,
    query_embed: Linear,
    act_embed0: Linear,
    act_embed2: Linear,
    prev_embed_w: Tensor, // [PREV_ACTIONS]
    prev_proj: Linear,
    blocks: Vec<Block>,
    pool_ln: LayerNorm,
    pool: Mha,
    policy_dot: Linear,
    value_head: Linear,
    action_table: Tensor, // [N, 8]
    d_model: usize,
    ent_feat: usize,
    q_feat: usize,
    prev_actions: usize,
    table_size: usize,
    dev: Device,
}

impl EntityPolicy {
    pub fn new(weights: &RawTensors, heads: usize) -> Result<Self, String> {
        ensure_single_thread_gemm();
        let dev = Device::Cpu;

        let (_, embed_w_shape) = weights.get("embed.weight").ok_or("missing embed.weight")?;
        if embed_w_shape.len() != 2 {
            return Err(format!("embed.weight must be 2D, got {embed_w_shape:?}"));
        }
        let d_model = embed_w_shape[0];
        let ent_feat = embed_w_shape[1];

        let (_, q_embed_w_shape) = weights.get("query_embed.weight").ok_or("missing query_embed.weight")?;
        let q_feat = q_embed_w_shape[1];

        let (_, prev_w_shape) = weights.get("prev_embed_w").ok_or("missing prev_embed_w")?;
        if prev_w_shape.len() != 1 {
            return Err(format!("prev_embed_w must be 1D, got {prev_w_shape:?}"));
        }
        let prev_actions = prev_w_shape[0];

        let (_, act_table_shape) = weights.get("action_table").ok_or("missing action_table")?;
        if act_table_shape.len() != 2 || act_table_shape[1] != ACT_FEAT {
            return Err(format!("action_table must be [N,{ACT_FEAT}], got {act_table_shape:?}"));
        }
        let table_size = act_table_shape[0];

        // layer count: scan for blocks.{i}.ln1.weight keys, must be dense 0..layers
        let mut block_ids: Vec<usize> = weights
            .keys()
            .filter_map(|k| k.strip_prefix("blocks.")?.strip_suffix(".ln1.weight")?.parse().ok())
            .collect();
        block_ids.sort_unstable();
        if block_ids.is_empty() {
            return Err("no transformer blocks found (no blocks.N.ln1.weight keys)".into());
        }
        if !block_ids.iter().enumerate().all(|(i, &id)| id == i) {
            return Err(format!("block indices must be dense 0..layers, got {block_ids:?}"));
        }

        let blocks = block_ids
            .iter()
            .map(|i| Block::new(weights, &format!("blocks.{i}"), d_model, heads, &dev))
            .collect::<Result<Vec<_>, _>>()?;

        let policy = Self {
            embed: linear(weights, "embed", &dev)?,
            query_embed: linear(weights, "query_embed", &dev)?,
            act_embed0: linear(weights, "act_embed.0", &dev)?,
            act_embed2: linear(weights, "act_embed.2", &dev)?,
            prev_embed_w: tensor(weights, "prev_embed_w", &dev)?,
            prev_proj: linear(weights, "prev_proj", &dev)?,
            blocks,
            pool_ln: layer_norm(weights, "pool_ln", &dev)?,
            pool: Mha::new(weights, "pool", d_model, heads, &dev)?,
            policy_dot: linear(weights, "policy_dot", &dev)?,
            value_head: linear(weights, "value_head", &dev)?,
            action_table: tensor(weights, "action_table", &dev)?,
            d_model,
            ent_feat,
            q_feat,
            prev_actions,
            table_size,
            dev,
        };

        // sanity: act_embed2 output must be ACT_EMB (policy_dot/prev_proj input dim)
        if policy.act_embed2.weight().dims()[0] != ACT_EMB {
            return Err(format!(
                "act_embed.2.weight out_dim must be {ACT_EMB}, got {}",
                policy.act_embed2.weight().dims()[0]
            ));
        }
        if policy.policy_dot.weight().dims()[0] != ACT_EMB {
            return Err(format!(
                "policy_dot.weight out_dim must be {ACT_EMB}, got {}",
                policy.policy_dot.weight().dims()[0]
            ));
        }

        Ok(policy)
    }

    fn act_embed(&self, x: &Tensor) -> Result<Tensor, String> {
        let e = |r: candle_core::Result<Tensor>| r.map_err(|e| e.to_string());
        let h = e(self.act_embed0.forward(x))?;
        let h = e(h.relu())?;
        e(self.act_embed2.forward(&h))
    }

    /// `ents [B,MAX_ENT,ENT_FEAT]`, `mask [B,MAX_ENT]` (true = ignore),
    /// `query [B,Q_FEAT]`, `prev [B,PREV_ACTIONS]` (indices into the action
    /// table). Returns (`logits` flat row-major `[B,table_size]`, `values [B]`).
    pub fn forward(
        &self,
        ents: &[f32],
        mask: &[bool],
        query: &[f32],
        prev: &[i64],
        b: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let dev = &self.dev;
        let e = |r: candle_core::Result<Tensor>| r.map_err(|err| err.to_string());

        let max_ent = mask.len() / b.max(1);
        if ents.len() != b * max_ent * self.ent_feat {
            return Err(format!(
                "ents length {} != b*max_ent*ent_feat ({}*{}*{})",
                ents.len(), b, max_ent, self.ent_feat
            ));
        }
        if mask.len() != b * max_ent {
            return Err(format!("mask length {} != b*max_ent ({}*{})", mask.len(), b, max_ent));
        }
        if query.len() != b * self.q_feat {
            return Err(format!("query length {} != b*q_feat ({}*{})", query.len(), b, self.q_feat));
        }
        if prev.len() != b * self.prev_actions {
            return Err(format!(
                "prev length {} != b*prev_actions ({}*{})",
                prev.len(), b, self.prev_actions
            ));
        }

        let entities = e(Tensor::from_vec(ents.to_vec(), (b, max_ent, self.ent_feat), dev))?;
        let query_t = e(Tensor::from_vec(query.to_vec(), (b, self.q_feat), dev))?;

        // additive key mask, shape [B,1,1,MAX_ENT], broadcast over heads/Tq.
        let mask_add: Vec<f32> = mask.iter().map(|&m| if m { MASK_NEG } else { 0.0 }).collect();
        let mask_add = e(Tensor::from_vec(mask_add, (b, 1, 1, max_ent), dev))?;

        // 1. embed entities, run pre-LN transformer blocks over the entity set
        let mut x = e(self.embed.forward(&entities))?; // [B,N,d]
        for blk in &self.blocks {
            x = blk.forward(&x, &mask_add)?;
        }

        // 2. query pooling: self/query token cross-attends over pool_ln(x)
        let q0 = e(self.query_embed.forward(&query_t))?;
        let q0 = e(q0.reshape((b, 1, self.d_model)))?;
        let kv = e(self.pool_ln.forward(&x))?;
        let pooled = self.pool.forward(&q0, &kv, &mask_add)?; // [B,1,d]
        let pooled = e(pooled.reshape((b, self.d_model)))?;

        // 3. prev-action pathway: gather rows from action_table, embed, weighted
        // sum with softmax(prev_embed_w), project into d_model, add to pooled.
        let prev_idx = e(Tensor::from_vec(prev.to_vec(), b * self.prev_actions, dev))?;
        let prev_rows = e(self.action_table.index_select(&prev_idx, 0))?; // [B*P,8]
        let prev_rows = e(prev_rows.reshape((b, self.prev_actions, ACT_FEAT)))?;
        let prev_e = self.act_embed(&prev_rows)?; // [B,P,32]

        let w = e(candle_nn::ops::softmax_last_dim(&self.prev_embed_w))?; // [P]
        let w = e(w.reshape((1, self.prev_actions, 1)))?;
        let prev_weighted = e(prev_e.broadcast_mul(&w))?;
        let prev_sum = e(prev_weighted.sum(1))?; // [B,32]
        let prev_add = e(self.prev_proj.forward(&prev_sum))?; // [B,d]
        let pooled = e(&pooled + prev_add)?;

        // 4. dot-product policy head + value head
        let player = e(self.policy_dot.forward(&pooled))?; // [B,32]
        let table_e = self.act_embed(&self.action_table)?; // [N,32]
        let table_e_t = e(table_e.t())?;
        let table_e_t = e(table_e_t.contiguous())?;
        let logits = e(player.matmul(&table_e_t))?; // [B,N]
        let value = e(self.value_head.forward(&pooled))?; // [B,1]

        let logits = e(logits.flatten_all())?;
        let logits = logits.to_vec1::<f32>().map_err(|err| err.to_string())?;
        let value = e(value.flatten_all())?;
        let value = value.to_vec1::<f32>().map_err(|err| err.to_string())?;

        let _ = self.table_size; // reserved for future validation hooks
        Ok((logits, value))
    }
}
