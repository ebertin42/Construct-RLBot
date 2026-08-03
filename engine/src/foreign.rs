//! Foreign (ported community-bot) opponent primitives: a candle MLP with LeakyReLU
//! plus (assembled in a later task) the obs + action-table + prev-action wrapper.
//! Kept separate from policy.rs (our v0 MLP) because the activation (LeakyReLU) and
//! the I/O contract differ. Weight-name contract: `net.{0,2,4,...}.{weight,bias}`
//! (torch Sequential of Linear + LeakyReLU), matching Immortal's jit.pt.
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use std::collections::HashMap;

#[inline]
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0
}

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
    /// Necto (~Diamond). Shares the EARL trunk with Nexto but has its OWN
    /// observation (ball-first entities, pos+vel relative with no heading
    /// rotation, blue-keyed team flags, stateful boost/demo timers) and a
    /// multi-discrete head. See obs_necto.rs.
    Necto,
    /// Element (RLMarlbot, by Rangler): AdvancedObs-107 (its CustomObs is
    /// functionally identical) -> 5x256 ReLU MLP -> 5 categorical + 3 Bernoulli
    /// heads -> controls-8 directly (no action table). Its scripted Speedflip
    /// kickoff is NOT ported -- the network only.
    Element,
}

impl ForeignKind {
    /// Immortal and Element zero YAW while the jump button is held
    /// (`yaw = 0 if action[5] > 0 else action[3]` in their bot.py); Nexto and
    /// Necto pass yaw through unchanged. Missing this makes an aerial bot
    /// dodge in the wrong direction.
    pub fn zero_yaw_on_jump(self) -> bool {
        matches!(self, Self::Immortal | Self::Element)
    }

    /// True for the bots whose obs is AdvancedObs-107, a FIXED width with room
    /// for exactly one other car. Above 1v1 they are fed a TRUNCATED obs
    /// showing only the opponent nearest the ball (teammates and the remaining
    /// opponents are invisible to them) -- see
    /// `obs_advanced::build_advanced_obs_one_other`. Nexto and Necto are EARL
    /// attention models that size their obs from `state.cars.len()`, so they
    /// see the whole arena and this is false for them.
    ///
    /// At 1v1 truncation is a no-op (there is exactly one other car and it is
    /// the enemy), so this only distinguishes behaviour in team arenas.
    /// Callers use it to label a slot as a DEGRADED LOCAL VARIANT rather than
    /// to refuse it: until 2026-07-26 these kinds were refused outright at
    /// mode > 1.
    pub fn truncates_obs(self) -> bool {
        matches!(self, Self::Immortal | Self::Element)
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "immortal" => Some(Self::Immortal),
            "nexto" => Some(Self::Nexto),
            "necto" => Some(Self::Necto),
            "element" => Some(Self::Element),
            _ => None,
        }
    }
}

/// The per-bot network + observation pathway.
enum Backend {
    Immortal(ForeignMlp),
    /// Nexto: EARL trunk + dot head, Nexto's observation.
    Nexto(crate::nexto::NextoNet),
    /// Necto: EARL trunk + multi-discrete head, and its OWN observation with
    /// per-arena stateful timers.
    Necto(crate::nexto::NextoNet),
    Element(crate::element::ElementNet),
}

/// A ported community bot driving one or more cars. Holds the net, its action
/// table, and the per-car previous action (Immortal feeds the last chosen
/// controls-8 back into its obs; it starts at zeros each episode).
pub struct ForeignPolicy {
    kind: ForeignKind,
    backend: Backend,
    table: Vec<[f32; 8]>,
    /// Difficulty handicap: recompute a decision only every `decision_period`
    /// calls, holding the previous controls in between. 1 = full strength.
    /// Every ported bot outclasses us (all four score 96-0 against the champion),
    /// so a tunable weakening is the only way to get a contestable teacher --
    /// a 0% win rate saturates the win-probability potential and kills the
    /// gradient. Reaction rate is the cleanest knob: it degrades skill smoothly
    /// without touching the policy or its observation.
    decision_period: u32,
    /// How many of an arena's ORANGE cars this bot drives (1..=team size). The
    /// rest of the orange team is driven by the LEARNER's own policy, which
    /// makes a team foreign arena contested by construction.
    ///
    /// This exists because MEASUREMENT showed `decision_period` has almost no
    /// authority above 1v1. Same net (ck_001439180800), reward_v0 tape, 8
    /// matches/mode vs a FULL nexto team: 1v1 us 88:5 at p12, but 3v3 us 1:25 at
    /// p12, and the 3v3 goal share only creeps 0.092 -> 0.256 -> 0.314 -> 0.371
    /// across p12/p24/p48/p96. At ONE DECISION EVERY 6.4 SECONDS a full nexto
    /// team still out-scores us. Car count is the knob that works at m > 1.
    ///
    /// Defaults to `u32::MAX` == "the whole orange team", which is exactly the
    /// pre-2026-07-26 behaviour (the override loop clamps to the team size).
    foreign_cars: u32,
    /// Per-car countdown + last controls, for the handicap.
    hold: HashMap<u64, (u32, [f32; 8])>,
    /// Necto only: per-ARENA stateful timers (boost respawn + demo), keyed by the
    /// arena part of the decide key. Necto carries these across frames.
    necto_timers: std::collections::HashMap<u64, crate::obs_necto::NectoTimers>,
    /// Keyed by a GLOBALLY unique id, not car id: RocketSim numbers cars from 1
    /// within each arena, so ids collide across arenas and a car-id key would
    /// share one previous-action entry between every arena in the slot.
    prev: HashMap<u64, [f32; 8]>,
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
            ForeignKind::Nexto => {
                let table = crate::actions::make_lookup_table();
                (Backend::Nexto(crate::nexto::NextoNet::new(w, &table)?), table)
            }
            ForeignKind::Necto => {
                let table = crate::actions::make_lookup_table();
                (Backend::Necto(crate::nexto::NextoNet::new(w, &table)?), table)
            }
            ForeignKind::Element => (
                // Element emits controls directly; the table stays empty.
                Backend::Element(crate::element::ElementNet::new(
                    w, crate::obs_advanced::ADV_OBS_SIZE)?),
                Vec::new(),
            ),
        };
        Ok(Self {
            kind,
            backend,
            table,
            necto_timers: std::collections::HashMap::new(),
            decision_period: 1,
            foreign_cars: u32::MAX,
            hold: HashMap::new(),
            prev: HashMap::new(),
        })
    }

    pub fn kind(&self) -> ForeignKind {
        self.kind
    }

    /// Drive only the first `n` orange cars of each arena (>=1). `u32::MAX` (the
    /// default) means the whole orange team. See the `foreign_cars` field.
    pub fn set_foreign_cars(&mut self, n: u32) {
        self.foreign_cars = n.max(1);
    }

    /// How many orange cars of an arena with `orange` of them this bot drives.
    pub fn cars_in_arena(&self, orange: usize) -> usize {
        (self.foreign_cars as usize).min(orange)
    }

    /// Handicap the bot by making it react every `n` decisions (>=1). At n=2 it
    /// acts at half our decision rate, and so on.
    pub fn set_decision_period(&mut self, n: u32) {
        self.decision_period = n.max(1);
    }

    pub fn decision_period(&self) -> u32 {
        self.decision_period
    }

    /// Clear per-car previous actions. Call on episode reset so the obs restarts
    /// from the zero action (matching the reference bot's `np.zeros(8)`).
    pub fn reset(&mut self) {
        self.prev.clear();
        self.hold.clear();
    }

    /// The RAW controls this bot last chose for `key`, before the jump/yaw
    /// override is applied.
    ///
    /// `decide` returns the OVERRIDDEN controls (yaw zeroed while jumping for
    /// kinds that need it) but stores the RAW choice here, because the bot feeds
    /// its raw action back into its own observation. A teacher LABEL must be the
    /// raw choice too — that is the action the bot actually selected, and the
    /// override is an actuation detail applied afterwards. The override is also
    /// lossy (it destroys yaw), so it cannot be inverted from the return value;
    /// this accessor is the only way to recover the label.
    pub fn prev_of(&self, key: u64) -> [f32; 8] {
        *self.prev.get(&key).unwrap_or(&[0.0; 8])
    }

    /// Overwrite one car's stored previous action.
    ///
    /// EXISTS FOR TEACHER LABELLING, and the distinction it enables is the whole
    /// point. `decide` ends by storing its OWN chosen controls as the next
    /// previous action, which is right when the bot is driving: its obs then
    /// reflects what it actually did.
    ///
    /// When the bot is used as a TEACHER over states produced by a DIFFERENT
    /// policy, that is wrong. The car physically executed the student's controls,
    /// so the teacher's observation must carry the student's last action, not the
    /// teacher's counterfactual one. Letting `decide` feed itself would answer
    /// "what would this bot do if it had been driving all along" — a question
    /// about a trajectory that never occurred.
    ///
    /// Call this after each step with the controls the car ACTUALLY executed.
    pub fn set_prev(&mut self, key: u64, controls: [f32; 8]) {
        self.prev.insert(key, controls);
    }

    /// Clear ONE car's previous action. Used at an episode boundary: a slot may
    /// drive several arenas, so a blanket `reset()` would wrongly clear cars whose
    /// episodes are still running.
    pub fn reset_car(&mut self, key: u64) {
        self.prev.remove(&key);
        self.hold.remove(&key);
        // Necto's boost-respawn and demo clocks are per ARENA and are carried
        // ACROSS frames, so an episode boundary must clear them too -- otherwise
        // the fresh episode inherits the previous one's countdowns (and, since
        // 2026-07-26, its cached `last_tick`, which the rebuilt-arena case can
        // reuse). Removal is by the arena half of the key and is idempotent, so
        // it is correct to call once per car of the arena as the collect loop does.
        //
        // THIS MATCHES THE REFERENCE and it BREAKS MEASUREMENT CONTINUITY. Both
        // halves matter:
        //   * `NectoObsBuilder.reset()` zeroes both clocks
        //     (deploy/external/necto/necto_obs.py:32-34: `demo_timers =
        //     Counter()`, `boost_timers = np.zeros(...)`), so the pre-2026-07-26
        //     engine, which leaked them across resets, was the one that was wrong.
        //   * necto is therefore NOT byte-identical to the pre-v9 build across an
        //     episode boundary, and it is the ONLY foreign path that changed.
        //     MEASURED (1v1 necto, reward_v0, `setarch -R`): with ZERO resets the
        //     two builds are byte-identical 2/2; with resets they diverge at the
        //     first step AFTER the first termination and stay diverged (pre-v9
        //     rew_sum in {-209.88, -190.97, -186.09, -178.16} over 6 runs at
        //     36-38 terminations; v9 -154.3225 at 33, 4/4 identical). nexto,
        //     element and immortal are unchanged (overlapping value sets), and
        //     self-play / native-opponent collects are byte-identical 3/3.
        // Consequence for anyone reading a number: EVERY necto row measured
        // before 2026-07-26 -- scripts/bench_foreign.py, scripts/bench_ladder.py,
        // scripts/version_ladder.py, and the L3 rung "necto 0.615 at p4" -- was
        // scored on the leaking-timer build and is NOT comparable to anything
        // measured after the v9 wheel is installed over
        // python/construct/_engine.abi3.so. Re-measure necto's rungs (the M0
        // seeding pass) on the installed v9 instrument before trusting them.
        // If a strictly non-regressive variant is ever needed to reproduce an old
        // necto bench, clearing ONLY `last_tick` (which the rebuilt-arena case
        // needs) without zeroing the boost/demo clocks reproduces the old numbers.
        self.necto_timers.remove(&(key >> 32));
    }

    /// One decision for `state.cars[car_idx]`: build the bot's own obs using its
    /// stored previous action, forward, argmax, look up controls-8, and store
    /// those controls as the next previous action.
    pub fn decide(&mut self, state: &rocketsim_rs::GameState, car_idx: usize, key: u64)
        -> [f32; 8]
    {
        // Handicap: hold the previous controls until the countdown expires. The
        // bot's own prev-action state is left untouched, so when it does think it
        // sees exactly what the reference would.
        if self.decision_period > 1 {
            if let Some((left, held)) = self.hold.get_mut(&key) {
                if *left > 0 {
                    *left -= 1;
                    return *held;
                }
            }
        }
        let prev = *self.prev.get(&key).unwrap_or(&[0.0; 8]);
        let controls = match &self.backend {
            Backend::Immortal(mlp) => {
                // Truncated obs: AdvancedObs-107 has room for exactly ONE other
                // car, so above 1v1 we show the opponent nearest the ball and
                // hide the rest. Byte-identical to the untruncated builder at
                // 1v1, so every existing 1v1 bench row stays comparable. See
                // obs_advanced::build_advanced_obs_one_other for the evidence.
                let mut obs = vec![0.0f32; crate::obs_advanced::ADV_OBS_SIZE];
                match crate::obs_advanced::nearest_opponent_to_ball(state, car_idx) {
                    Some(oi) => crate::obs_advanced::build_advanced_obs_one_other(
                        state, car_idx, oi, &prev, &mut obs,
                    ),
                    // No opponents at all (every enemy demoed / a malformed
                    // arena): idle rather than feed the net a zeroed enemy slot.
                    None => return [0.0; 8],
                }
                match mlp.forward(&obs, 1, crate::obs_advanced::ADV_OBS_SIZE) {
                    Ok(l) => self.table[argmax(&l)],
                    // A forward failure must not kill a rollout; do nothing instead.
                    Err(_) => return [0.0; 8],
                }
            }
            Backend::Nexto(net) => {
                use crate::obs_nexto::{build_nexto_obs, n_entities, NEXTO_KV, NEXTO_Q};
                let n_ent = n_entities(state.cars.len());
                let mut q = vec![0.0f32; NEXTO_Q];
                let mut kv = vec![0.0f32; n_ent * NEXTO_KV];
                let mut mask = vec![false; n_ent];
                build_nexto_obs(state, car_idx, &prev, &mut q, &mut kv, &mut mask);
                match net.forward_head(&q, &kv, &mask) {
                    Ok(crate::nexto::HeadOut::Dot(l)) => self.table[argmax(&l)],
                    _ => return [0.0; 8],
                }
            }
            Backend::Element(net) => {
                // CustomObs is functionally identical to AdvancedObs-107, and
                // gets the same near-ball truncation as Immortal above.
                let mut obs = vec![0.0f32; crate::obs_advanced::ADV_OBS_SIZE];
                match crate::obs_advanced::nearest_opponent_to_ball(state, car_idx) {
                    Some(oi) => crate::obs_advanced::build_advanced_obs_one_other(
                        state, car_idx, oi, &prev, &mut obs,
                    ),
                    None => return [0.0; 8],
                }
                match net.decide(&obs) {
                    Ok(c) => c,
                    Err(_) => return [0.0; 8],
                }
            }
            Backend::Necto(net) => {
                use crate::obs_necto::{build_necto_obs, n_entities, NectoTimers,
                                       NECTO_KV, NECTO_Q};
                let n_ent = n_entities(state.cars.len());
                let mut q = vec![0.0f32; NECTO_Q];
                let mut kv = vec![0.0f32; n_ent * NECTO_KV];
                let mut mask = vec![false; n_ent];
                // timers are per ARENA (the high half of the key), not per car
                let arena = key >> 32;
                let timers = self
                    .necto_timers
                    .entry(arena)
                    .or_insert_with(|| NectoTimers::new(state.cars.len()));
                build_necto_obs(state, car_idx, &prev, timers, &mut q, &mut kv, &mut mask);
                match net.forward_head(&q, &kv, &mask) {
                    // five multi-discrete heads decoded straight to controls
                    Ok(crate::nexto::HeadOut::MultiDiscrete(h)) => {
                        crate::nexto::decode_multi_discrete(&h)
                    }
                    _ => return [0.0; 8],
                }
            }
        };
        // The bot feeds its RAW chosen action back into its own observation, but
        // applies the jump/yaw override only when setting controls -- keep the two
        // separate or the obs drifts from the reference.
        self.prev.insert(key, controls);
        let mut out = controls;
        if self.kind.zero_yaw_on_jump() && out[5] > 0.0 {
            out[3] = 0.0;
        }
        if self.decision_period > 1 {
            self.hold.insert(key, (self.decision_period - 1, out));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Immortal-shaped state_dict, all zeros. The KIND is irrelevant to what is
    /// under test below -- `reset_car` operates on the timer MAP, not on the
    /// network -- and immortal is the only shape that can be fabricated without
    /// the unlicensed weights.
    fn zero_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        let mut w = HashMap::new();
        for (idx, out_dim, in_dim) in [
            (0usize, 512usize, 107usize), (2, 512, 512), (4, 512, 512), (6, 512, 512),
            (8, 512, 512), (10, 512, 512), (12, 126, 512),
        ] {
            w.insert(format!("net.{idx}.weight"),
                     (vec![0.0f32; out_dim * in_dim], vec![out_dim, in_dim]));
            w.insert(format!("net.{idx}.bias"), (vec![0.0f32; out_dim], vec![out_dim]));
        }
        w
    }

    /// TRIPWIRE for a behaviour change that INVALIDATES MEASUREMENTS, not just a
    /// unit test. Clearing necto's per-arena boost/demo clocks on an episode
    /// boundary matches `NectoObsBuilder.reset()` and is a fidelity fix, but it
    /// means every necto bench/ladder row measured before 2026-07-26 was scored
    /// on different behaviour (measured: identical with zero resets, divergent
    /// from the first step after the first reset). If this ever silently
    /// reverts, the necto rungs move again and nothing else says so.
    #[test]
    fn reset_car_clears_that_arenas_necto_timers_and_only_that_arenas() {
        use crate::obs_necto::NectoTimers;
        let key = |arena: u64, car: u64| (arena << 32) | car;
        let mut p = ForeignPolicy::new(&zero_weights(), ForeignKind::Immortal).unwrap();
        p.necto_timers.insert(0, NectoTimers::new(2));
        p.necto_timers.insert(7, NectoTimers::new(2));

        p.reset_car(key(0, 1));
        assert!(!p.necto_timers.contains_key(&0), "the reset arena's clocks must be cleared");
        assert!(p.necto_timers.contains_key(&7), "another arena's clocks must survive");

        // the collect loop calls this once per CAR of the arena, so it must be
        // idempotent -- a second call for car 2 of the same arena is normal.
        p.reset_car(key(0, 2));
        assert_eq!(p.necto_timers.len(), 1);
    }
}
