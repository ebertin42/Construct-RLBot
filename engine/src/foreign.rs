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

/// Which ported bot a foreign opponent slot runs. Selects the obs builder, the
/// action table and the MLP dims. Element/Nexto would extend this enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForeignKind {
    /// RLMarlbot's Immortal: AdvancedObs-107 -> 7xLinear/LeakyReLU(512) -> 126 logits
    /// -> argmax -> 126-row lookup -> controls-8. Deterministic (argmax), matching
    /// its deploy behaviour.
    Immortal,
    /// Nexto (~GC1) -- EARLPerceiver entity obs (q32/kv24/mask) -> 2 cross-attention
    /// blocks -> ControlsPredictorDot -> 90 logits -> argmax -> the 90-row table
    /// (byte-identical to our own `make_lookup_table`).
    Nexto,
    /// Necto (~Diamond): same architecture and action table as Nexto, weaker weights.
    Necto,
}

impl ForeignKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "immortal" => Some(Self::Immortal),
            "nexto" => Some(Self::Nexto),
            "necto" => Some(Self::Necto),
            _ => None,
        }
    }
}

/// The per-bot network + observation pathway.
enum Backend {
    Immortal(ForeignMlp),
    /// Nexto and Necto share this; only the weights differ.
    Earl(crate::nexto::NextoNet),
}

/// A ported community bot driving one or more cars. Holds the net, its action
/// table, and the per-car previous action (Immortal feeds the last chosen
/// controls-8 back into its obs; it starts at zeros each episode).
pub struct ForeignPolicy {
    kind: ForeignKind,
    backend: Backend,
    table: Vec<[f32; 8]>,
    prev: HashMap<u32, [f32; 8]>,
}

impl ForeignPolicy {
    pub fn new(
        w: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        kind: ForeignKind,
    ) -> Result<Self, String> {
        let (backend, table) = match kind {
            ForeignKind::Immortal => (
                Backend::Immortal(ForeignMlp::from_named(
                    w, crate::obs_advanced::ADV_OBS_SIZE, 512, 126, 6)?),
                crate::actions::make_immortal_table(),
            ),
            ForeignKind::Nexto | ForeignKind::Necto => {
                let table = crate::actions::make_lookup_table();
                (Backend::Earl(crate::nexto::NextoNet::new(w, &table)?), table)
            }
        };
        Ok(Self { kind, backend, table, prev: HashMap::new() })
    }

    pub fn kind(&self) -> ForeignKind {
        self.kind
    }

    /// Clear per-car previous actions. Call on episode reset so the obs restarts
    /// from the zero action (matching the reference bot's `np.zeros(8)`).
    pub fn reset(&mut self) {
        self.prev.clear();
    }

    /// Clear ONE car's previous action. Used at an episode boundary: a slot may
    /// drive several arenas, so a blanket `reset()` would wrongly clear cars whose
    /// episodes are still running.
    pub fn reset_car(&mut self, car_id: u32) {
        self.prev.remove(&car_id);
    }

    /// One decision for `state.cars[car_idx]`: build the bot's own obs using its
    /// stored previous action, forward, argmax, look up controls-8, and store
    /// those controls as the next previous action.
    pub fn decide(&mut self, state: &rocketsim_rs::GameState, car_idx: usize) -> [f32; 8] {
        let car_id = state.cars[car_idx].id;
        let prev = *self.prev.get(&car_id).unwrap_or(&[0.0; 8]);
        let logits = match &self.backend {
            Backend::Immortal(mlp) => {
                let mut obs = vec![0.0f32; crate::obs_advanced::ADV_OBS_SIZE];
                crate::obs_advanced::build_advanced_obs(state, car_idx, &prev, &mut obs);
                mlp.forward(&obs, 1, crate::obs_advanced::ADV_OBS_SIZE)
            }
            Backend::Earl(net) => {
                use crate::obs_nexto::{build_nexto_obs, n_entities, NEXTO_KV, NEXTO_Q};
                let n_ent = n_entities(state.cars.len());
                let mut q = vec![0.0f32; NEXTO_Q];
                let mut kv = vec![0.0f32; n_ent * NEXTO_KV];
                let mut mask = vec![false; n_ent];
                build_nexto_obs(state, car_idx, &prev, &mut q, &mut kv, &mut mask);
                net.forward(&q, &kv, &mask)
            }
        };
        let logits = match logits {
            Ok(l) => l,
            // A forward failure must not kill a rollout; fall back to "do nothing".
            Err(_) => return [0.0; 8],
        };
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        let controls = self.table[best];
        self.prev.insert(car_id, controls);
        controls
    }
}
