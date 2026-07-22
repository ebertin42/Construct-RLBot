//! Element (RLMarlbot, by Rangler) ported to candle.
//!
//! Architecture read from rlmarlbot/element/agent.py::Actor:
//! ```text
//! x = relu(fc1..fc5(x))                 5 x Linear(256), ReLU  (fc1 is 107->256)
//! cat = softmax(cat_heads(x).view(5,3)) 5 categorical heads of 3
//! ber = softmax(ber_heads(x).view(3,2)) 3 Bernoulli heads of 2
//! action = concat(argmax(cat), argmax(ber)) + [-1,-1,-1,-1,-1,0,0,0]
//! ```
//! Softmax is monotonic, so the argmax is taken on the logits and the softmax is
//! skipped.
//!
//! Element's OBSERVATION is `CustomObs`, which is functionally IDENTICAL to
//! Immortal's AdvancedObs-107 (same POS_STD/ANG_STD, same concatenation order,
//! same per-player block) -- verified by diffing the two shipped builders -- so
//! `obs_advanced::build_advanced_obs` is reused rather than duplicated.
//!
//! NOT ported: Element's scripted `Speedflip` kickoff sequence, which overrides
//! the net at kickoff. Our port is the network only, so it kicks off like the
//! model alone would. See docs/foreign-opponents.md.
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use std::collections::HashMap;

type Raw = HashMap<String, (Vec<f32>, Vec<usize>)>;

pub const ELEMENT_CATEGORICALS: usize = 5;
pub const ELEMENT_BERNOULLIS: usize = 3;
/// `bot.py::action_trans` -- the five categorical heads are shifted from {0,1,2}
/// to {-1,0,1}; the Bernoulli heads stay {0,1}.
const ACTION_TRANS: [f32; 8] = [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0];

fn e<T>(r: candle_core::Result<T>) -> Result<T, String> {
    r.map_err(|x| x.to_string())
}

fn linear(map: &Raw, prefix: &str, dev: &Device) -> Result<Linear, String> {
    let (w, ws) = map.get(&format!("{prefix}.weight"))
        .ok_or_else(|| format!("missing {prefix}.weight"))?;
    let (b, bs) = map.get(&format!("{prefix}.bias"))
        .ok_or_else(|| format!("missing {prefix}.bias"))?;
    if ws.len() != 2 {
        return Err(format!("{prefix}.weight must be 2-D, got {ws:?}"));
    }
    let wt = e(Tensor::from_vec(w.clone(), (ws[0], ws[1]), dev))?;
    let bt = e(Tensor::from_vec(b.clone(), bs[0], dev))?;
    Ok(Linear::new(wt, Some(bt)))
}

pub struct ElementNet {
    fc: Vec<Linear>,
    cat_heads: Linear,
    ber_heads: Linear,
}

impl ElementNet {
    pub fn new(map: &Raw, obs_size: usize) -> Result<Self, String> {
        crate::policy::ensure_single_thread_gemm();
        let dev = Device::Cpu;
        let fc = (1..=5)
            .map(|i| linear(map, &format!("fc{i}"), &dev))
            .collect::<Result<Vec<_>, _>>()?;
        // shape guard: a mismatched obs width here would silently feed garbage
        let w0 = &map["fc1.weight"].1;
        if w0[1] != obs_size {
            return Err(format!("fc1.weight in-dim {} != obs size {obs_size}", w0[1]));
        }
        let cat_heads = linear(map, "cat_heads", &dev)?;
        let ber_heads = linear(map, "ber_heads", &dev)?;
        if map["cat_heads.weight"].1[0] != 3 * ELEMENT_CATEGORICALS {
            return Err("cat_heads must emit 3 * 5 logits".into());
        }
        if map["ber_heads.weight"].1[0] != 2 * ELEMENT_BERNOULLIS {
            return Err("ber_heads must emit 2 * 3 logits".into());
        }
        Ok(Self { fc, cat_heads, ber_heads })
    }

    /// Returns the raw controls-8 action (post `action_trans`, pre any
    /// jump/yaw override, which the caller applies).
    pub fn decide(&self, obs: &[f32]) -> Result<[f32; 8], String> {
        let dev = Device::Cpu;
        let mut x = e(Tensor::from_vec(obs.to_vec(), (1, obs.len()), &dev))?;
        for l in &self.fc {
            x = e(e(l.forward(&x))?.relu())?;
        }
        let cat = e(e(self.cat_heads.forward(&x))?.flatten_all())?;
        let cat = e(cat.to_vec1::<f32>())?;
        let ber = e(e(self.ber_heads.forward(&x))?.flatten_all())?;
        let ber = e(ber.to_vec1::<f32>())?;

        let am = |v: &[f32]| {
            v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            }).0
        };
        let mut out = [0.0f32; 8];
        // cat_heads is viewed as (categoricals, 3): head h occupies [3h, 3h+3)
        for h in 0..ELEMENT_CATEGORICALS {
            out[h] = am(&cat[3 * h..3 * h + 3]) as f32 + ACTION_TRANS[h];
        }
        // ber_heads is viewed as (bernoullis, 2): head h occupies [2h, 2h+2)
        for h in 0..ELEMENT_BERNOULLIS {
            let i = ELEMENT_CATEGORICALS + h;
            out[i] = am(&ber[2 * h..2 * h + 2]) as f32 + ACTION_TRANS[i];
        }
        Ok(out)
    }
}
