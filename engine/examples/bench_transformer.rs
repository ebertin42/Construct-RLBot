// TASK T1 MICROBENCH (entity-transformer plan, docs/superpowers/plans/
// 2026-07-16-entity-transformer-obs-v1.md). Minimal pre-LN transformer
// encoder forward pass in candle-core/candle-nn 0.11, random weights, f32,
// CPU, single-threaded gemm. Measures µs/forward across dims/batch configs
// to gate the launch net size for the entity policy (obs v1).
//
// Shapes mirror the plan's net I/O contract:
//   forward(entities [B,17,26], mask [B,17], query [B,64], prev_actions [B,5])
//     -> (logits [B,92], value [B,1])
// mask is omitted here (bench worst case = all 17 entities attended, i.e.
// no masking-induced sparsity to save compute). prev_actions are embedded
// via the same 8->d->32 action-embedding MLP and summed into the pooled
// embedding with fixed (non-learned-for-bench) position weights -- this
// bench only needs the FLOP/memory shape to be right, not trained weights.
//
// Run (niced -- a corpus build may be running on this box):
//   nice -n 15 cargo run --release --example bench_transformer -p construct-engine
use candle_core::{DType, Device, Result as CResult, Tensor};
use candle_nn::{LayerNorm, Linear, Module};
use std::time::Instant;

const MAX_ENT: usize = 17;
const ENT_FEAT: usize = 26;
const Q_FEAT: usize = 64;
const PREV_ACTIONS: usize = 5;
const ACT_FEAT: usize = 8;
const TABLE_SIZE: usize = 92;
const DOT_DIM: usize = 32;

// Reference remote sps with the current 94-float MLP policy at 192 arenas
// (from .superpowers/sdd notes / two-box-training memory). Used only to
// contextualize the projected numbers below -- not a hard input to the gate.
const MLP_REFERENCE_SPS: f64 = 101_000.0;

/// candle's CPU matmul spins a process-wide rayon pool sized by
/// RAYON_NUM_THREADS. Force single-threaded gemm for a clean per-forward
/// timing (mirrors construct_engine::policy::ensure_single_thread_gemm,
/// duplicated locally so this example has no dependency on policy.rs
/// internals beyond what's already `pub`).
fn ensure_single_thread_gemm() {
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }
}

fn rand_linear(in_dim: usize, out_dim: usize, dev: &Device) -> CResult<Linear> {
    let bound = 1.0 / (in_dim as f64).sqrt();
    let w = Tensor::rand(-bound as f32, bound as f32, (out_dim, in_dim), dev)?;
    let b = Tensor::rand(-bound as f32, bound as f32, out_dim, dev)?;
    Ok(Linear::new(w, Some(b)))
}

fn rand_layer_norm(d: usize, dev: &Device) -> CResult<LayerNorm> {
    let w = Tensor::ones(d, DType::F32, dev)?;
    let b = Tensor::zeros(d, DType::F32, dev)?;
    Ok(LayerNorm::new(w, b, 1e-5))
}

/// Standard multi-head attention via primitives (no flash-attn -- seq is
/// tiny, 17 or 18 tokens). q_in: [B,Nq,d], kv_in: [B,Nk,d] -> [B,Nq,d].
struct Mha {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    heads: usize,
    head_dim: usize,
}

impl Mha {
    fn new(d: usize, heads: usize, dev: &Device) -> CResult<Self> {
        assert_eq!(d % heads, 0, "d_model must be divisible by heads");
        Ok(Self {
            wq: rand_linear(d, d, dev)?,
            wk: rand_linear(d, d, dev)?,
            wv: rand_linear(d, d, dev)?,
            wo: rand_linear(d, d, dev)?,
            heads,
            head_dim: d / heads,
        })
    }

    fn forward(&self, q_in: &Tensor, kv_in: &Tensor) -> CResult<Tensor> {
        let qd = q_in.dims();
        let kd = kv_in.dims();
        let (b, nq) = (qd[0], qd[1]);
        let nk = kd[1];

        let q = self.wq.forward(q_in)?;
        let k = self.wk.forward(kv_in)?;
        let v = self.wv.forward(kv_in)?;

        let q = q
            .reshape((b, nq, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // [B,h,Nq,hd]
        let k = k
            .reshape((b, nk, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // [B,h,Nk,hd]
        let v = v
            .reshape((b, nk, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // [B,h,Nk,hd]

        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?; // [B,h,Nq,Nk]
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?; // [B,h,Nq,hd]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, nq, self.heads * self.head_dim))?;
        self.wo.forward(&out)
    }
}

struct Block {
    ln1: LayerNorm,
    attn: Mha,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

impl Block {
    fn new(d: usize, heads: usize, ff: usize, dev: &Device) -> CResult<Self> {
        Ok(Self {
            ln1: rand_layer_norm(d, dev)?,
            attn: Mha::new(d, heads, dev)?,
            ln2: rand_layer_norm(d, dev)?,
            ff1: rand_linear(d, ff, dev)?,
            ff2: rand_linear(ff, d, dev)?,
        })
    }

    // pre-LN: x = x + MHA(LN(x)); x = x + FF(LN(x))
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h, &h)?;
        let x = (x + h)?;
        let h = self.ln2.forward(&x)?;
        let h = self.ff1.forward(&h)?.gelu()?;
        let h = self.ff2.forward(&h)?;
        x + h
    }
}

struct EntityNet {
    embed: Linear,       // 26 -> d
    query_embed: Linear, // 64 -> d
    blocks: Vec<Block>,
    pool: Mha, // query cross-attends over entity outputs
    act_embed1: Linear, // 8 -> d
    act_embed2: Linear, // d -> 32
    policy_dot: Linear, // d -> 32 (player embedding)
    value_head: Linear, // d -> 1
    action_table: Tensor, // [92, 8] fixed random "table"
    d: usize,
}

impl EntityNet {
    fn new(d: usize, layers: usize, heads: usize, ff: usize, dev: &Device) -> CResult<Self> {
        let blocks = (0..layers)
            .map(|_| Block::new(d, heads, ff, dev))
            .collect::<CResult<Vec<_>>>()?;
        let action_table = Tensor::rand(-1.0f32, 1.0f32, (TABLE_SIZE, ACT_FEAT), dev)?;
        Ok(Self {
            embed: rand_linear(ENT_FEAT, d, dev)?,
            query_embed: rand_linear(Q_FEAT, d, dev)?,
            blocks,
            pool: Mha::new(d, heads, dev)?,
            act_embed1: rand_linear(ACT_FEAT, d, dev)?,
            act_embed2: rand_linear(d, DOT_DIM, dev)?,
            policy_dot: rand_linear(d, DOT_DIM, dev)?,
            value_head: rand_linear(d, 1, dev)?,
            action_table,
            d,
        })
    }

    /// entities [B,17,26], query [B,64], prev [B,5,8] (already-looked-up
    /// prev-action feature rows -- the lookup itself is O(B*5) scalar work,
    /// irrelevant to the candle-forward timing this bench targets).
    fn forward(&self, entities: &Tensor, query: &Tensor, prev: &Tensor) -> CResult<(Tensor, Tensor)> {
        let b = entities.dims()[0];

        // 1. embed entities + query
        let mut x = self.embed.forward(entities)?; // [B,17,d]
        let q0 = self.query_embed.forward(query)?.reshape((b, 1, self.d))?; // [B,1,d]

        // 2. L pre-LN transformer blocks over entity tokens
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }

        // 3. query pooling: self/query token cross-attends over block outputs
        let pooled = self.pool.forward(&q0, &x)?; // [B,1,d]

        // prev-action embeddings summed with fixed position weights, added
        // into the pooled embedding (matches plan: "summed with learned
        // position weights ... fed as prev_actions").
        let prev_h = self.act_embed1.forward(prev)?.gelu()?; // [B,5,d]
        let pos_w: Vec<f32> = (0..PREV_ACTIONS).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        let pos_w = Tensor::from_vec(pos_w, (1, PREV_ACTIONS, 1), entities.device())?;
        let prev_h = prev_h.broadcast_mul(&pos_w)?.sum(1)?.reshape((b, 1, self.d))?; // [B,1,d]
        let pooled = (pooled + prev_h)?;
        let player_emb = pooled.reshape((b, self.d))?;

        // 4. action-embedding dot-product policy head
        let act_emb = self
            .act_embed2
            .forward(&self.act_embed1.forward(&self.action_table)?.gelu()?)?; // [92,32]
        let player_dot = self.policy_dot.forward(&player_emb)?; // [B,32]
        let logits = player_dot.matmul(&act_emb.t()?)?; // [B,92]

        // 5. value head
        let value = self.value_head.forward(&player_emb)?; // [B,1]

        Ok((logits, value))
    }
}

struct Config {
    name: &'static str,
    d: usize,
    layers: usize,
    heads: usize,
    ff: usize,
}

const CONFIGS: &[Config] = &[
    Config { name: "128/2/4/512", d: 128, layers: 2, heads: 4, ff: 512 },
    Config { name: "192/3/6/768", d: 192, layers: 3, heads: 6, ff: 768 },
    Config { name: "256/4/8/1024", d: 256, layers: 4, heads: 8, ff: 1024 },
];

const BATCH_SIZES: &[usize] = &[48, 96, 192];
const WARMUP: usize = 10;
const ITERS: usize = 100;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn main() -> CResult<()> {
    ensure_single_thread_gemm();
    let dev = Device::Cpu;

    println!(
        "candle transformer microbench -- entity-transformer T1 (RAYON_NUM_THREADS={})",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_default()
    );
    println!("seq={MAX_ENT} ent_feat={ENT_FEAT} q_feat={Q_FEAT} prev_actions={PREV_ACTIONS} table={TABLE_SIZE}x{ACT_FEAT}");
    println!("MLP reference (remote, 192 arenas): ~{:.0} sps\n", MLP_REFERENCE_SPS);

    println!("| config | batch | median us/forward | projected sps (B/t_forward) |");
    println!("|---|---|---|---|");

    let mut rows: Vec<(String, usize, f64, f64)> = Vec::new();

    for cfg in CONFIGS {
        let net = EntityNet::new(cfg.d, cfg.layers, cfg.heads, cfg.ff, &dev)?;
        for &b in BATCH_SIZES {
            let entities = Tensor::rand(-1.0f32, 1.0f32, (b, MAX_ENT, ENT_FEAT), &dev)?;
            let query = Tensor::rand(-1.0f32, 1.0f32, (b, Q_FEAT), &dev)?;
            let prev = Tensor::rand(-1.0f32, 1.0f32, (b, PREV_ACTIONS, ACT_FEAT), &dev)?;

            for _ in 0..WARMUP {
                let (logits, value) = net.forward(&entities, &query, &prev)?;
                let _ = logits.sum_all()?.to_scalar::<f32>()?;
                let _ = value.sum_all()?.to_scalar::<f32>()?;
            }

            let mut samples = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let t0 = Instant::now();
                let (logits, value) = net.forward(&entities, &query, &prev)?;
                // force materialization so lazy graph doesn't skip work
                let _ = logits.sum_all()?.to_scalar::<f32>()?;
                let _ = value.sum_all()?.to_scalar::<f32>()?;
                samples.push(t0.elapsed().as_secs_f64() * 1e6); // us
            }

            let med_us = median(samples);
            let t_forward_s = med_us * 1e-6;
            let projected_sps = b as f64 / t_forward_s;

            println!(
                "| {} | {} | {:.1} | {:.0} |",
                cfg.name, b, med_us, projected_sps
            );
            rows.push((cfg.name.to_string(), b, med_us, projected_sps));
        }
    }

    println!("\nSUMMARY (1 forward = 1 env step of B agents; single-threaded gemm)");
    for (name, b, med_us, sps) in &rows {
        println!(
            "  {:>14} B={:<4} {:>8.1} us/forward  -> projected {:>9.0} sps (1 thread)",
            name, b, med_us, sps
        );
    }

    Ok(())
}
