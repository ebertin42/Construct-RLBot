use crate::{
    actions,
    curriculum::CurriculumConfig,
    episode::{ActionTableKind, EpisodeArena, ObsMode, StepFlags},
    obs::OBS_SIZE,
    obs_v1::{ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT},
    policy::{LayerWeights, MlpPolicy, PolicyWeights},
    policy_v1::EntityPolicy,
    reward::RewardConfig,
    sampler::{sample_categorical, Pcg32},
    schema::Schema,
};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Raw state-dict arrays as they arrive from Python: `name -> (flat
/// row-major f32 data, shape)`. V0 parses this into `PolicyWeights`
/// (`parse_state_dict`); v1 hands it to `EntityPolicy::new` unparsed (that
/// constructor does its own name/shape validation).
pub type RawStateDict = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// A policy state dict tagged by net family. `MultiEngine::set_weights` /
/// `set_opponents` reject a variant that doesn't match the engine's
/// `ObsMode` (schema version), so workers can assume the match.
pub enum NetWeights {
    V0(PolicyWeights),
    V1 { raw: RawStateDict, heads: usize },
    /// A ported community bot (see `foreign.rs`): its own obs builder, MLP and
    /// action table. Lives in a SEPARATE slot space from `set_opponents`
    /// (`set_foreign_opponents`), addressed by `k <= -2` in a collect assignment.
    /// `cars` = how many of an arena's ORANGE cars this bot drives (E4);
    /// `u32::MAX` means the whole orange team, which is the historical
    /// behaviour and the default lib.rs fills in.
    Foreign {
        raw: RawStateDict,
        kind: crate::foreign::ForeignKind,
        period: u32,
        cars: u32,
    },
}

enum Cmd {
    Reset,
    Step(Vec<i64>),
    Debug { local_idx: usize },
    SetWeights(Arc<NetWeights>),
    SetOpponents(Arc<Vec<NetWeights>>),
    /// Foreign (ported-bot) opponent slots. Separate slot space from
    /// `SetOpponents`; a Collect assignment addresses slot `f` as `-(f) - 2`.
    SetForeignOpponents(Arc<Vec<NetWeights>>),
    // `assignment` is the FULL (global, length num_arenas) opponent assignment;
    // each worker slices its own `[global_base..global_base+count)` range out of
    // it (see the Cmd::Collect arm). Legacy calls (Python `arena_opponents=None`)
    // are materialized by lib.rs into `vec![-1; num_arenas]` before this is ever
    // constructed — there is no separate "no assignment" variant, so the legacy
    // path and the opponent path run through literally the same code (byte-
    // identity regression test pins this).
    Collect { steps: usize, assignment: Arc<Vec<i32>> },
    /// Drain each arena's per-term reward counters (E9). Read-and-reset, so a
    /// caller gets a per-interval delta rather than a running total.
    RewardTerms,
    Shutdown,
}

/// Plain-Vecs rollout buffer produced by one worker's `Cmd::Collect` (and, after
/// gather, by `MultiEngine::collect`). Round-major layout: for shape `(T, N, ...)`
/// fields, round `t`'s data occupies `[t*N*k .. (t+1)*N*k)`. `last_values` is the
/// single post-rollout bootstrap value per agent (no T dimension). `pub`/`pub`
/// fields (not just the struct) because `MultiEngine::collect` returns this across
/// the module boundary into lib.rs, which reads each field directly to build numpy
/// arrays — same reasoning as `policy::PolicyWeights`/`LayerWeights`.
pub struct CollectOut {
    pub obs: Vec<f32>,
    // --- v1 (entity) obs buffers; empty in v0 mode, `obs` empty in v1 ---
    /// `[T, N, MAX_ENT, ENT_FEAT]` flattened.
    pub ents: Vec<f32>,
    /// `[T, N, MAX_ENT]` flattened (true = masked/absent).
    pub mask: Vec<bool>,
    /// `[T, N, Q_FEAT]` flattened.
    pub query: Vec<f32>,
    /// `[T, N, PREV_ACTIONS]` flattened (action-table indices).
    pub prev: Vec<i64>,
    /// Kickstart-teacher input `[T, N, OBS_SIZE]`; populated only when the
    /// engine was built with `emit_v0_obs=true` (v1 mode), else empty.
    pub obs_v0: Vec<f32>,
    pub actions: Vec<i64>,
    pub logprobs: Vec<f32>,
    pub values: Vec<f32>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub final_values: Vec<f32>,
    pub last_values: Vec<f32>,
    // Number of LEARNER rows (`agents` param to `zeros` below) — every buffer
    // above is shaped by this count, not by the raw agent count. Legacy (all
    // self-play) calls have `learner_agents == total agents`; opponent arenas
    // shrink it (self-play arenas contribute all agents, opponent arenas
    // contribute only their blue agents — see Cmd::Collect's `learner_idx`).
    pub learner_agents: usize,
    /// `(N,)` team size (1/2/3) of the arena each learner ROW came from (E5).
    /// Empty in v0 mode (v0 collects predate mixed team sizes as a trained
    /// regime and nothing consumes the label there).
    ///
    /// This is the label `ppo.py`'s per-group advantage standardisation keys
    /// on, and it MUST come from here rather than being reconstructed in
    /// Python: the worker split is `(num_arenas - assigned) / (threads - t)`
    /// with `threads` auto-detected from `available_parallelism`, so a Python
    /// reconstruction would silently disagree on some machines and mis-scale
    /// every gradient without producing an error anywhere.
    pub learner_team_size: Vec<i8>,
    /// Rows put through OUR policy's batched forward per round: learner rows
    /// plus the "mirror" cars of a partial foreign team (E4), whose experience
    /// is discarded. Equals `learner_agents` whenever no arena runs a partial
    /// foreign team.
    ///
    /// Exposed because it is the only direct evidence that the mirrors are
    /// being PLAYED by the policy rather than acting uniformly at random (which
    /// is what an unforwarded, unoverridden car does -- it samples from an
    /// all-zero logits row), and because it is the compute the run actually
    /// pays for (R5).
    pub forward_rows: usize,
}

impl CollectOut {
    fn zeros(steps: usize, agents: usize, obs_dim: usize) -> Self {
        CollectOut {
            obs: vec![0.0; steps * agents * obs_dim],
            ents: vec![],
            mask: vec![],
            query: vec![],
            prev: vec![],
            obs_v0: vec![],
            actions: vec![0; steps * agents],
            logprobs: vec![0.0; steps * agents],
            values: vec![0.0; steps * agents],
            rewards: vec![0.0; steps * agents],
            terminated: vec![false; steps * agents],
            truncated: vec![false; steps * agents],
            final_values: vec![0.0; steps * agents],
            last_values: vec![0.0; agents],
            learner_agents: agents,
            learner_team_size: vec![],
            forward_rows: agents,
        }
    }

    /// V1 counterpart of `zeros`: entity buffers sized, `obs` unused.
    fn zeros_v1(steps: usize, agents: usize, emit_v0_obs: bool) -> Self {
        CollectOut {
            obs: vec![],
            ents: vec![0.0; steps * agents * MAX_ENT * ENT_FEAT],
            mask: vec![false; steps * agents * MAX_ENT],
            query: vec![0.0; steps * agents * Q_FEAT],
            prev: vec![0; steps * agents * PREV_ACTIONS],
            obs_v0: if emit_v0_obs { vec![0.0; steps * agents * OBS_SIZE] } else { vec![] },
            actions: vec![0; steps * agents],
            logprobs: vec![0.0; steps * agents],
            values: vec![0.0; steps * agents],
            rewards: vec![0.0; steps * agents],
            terminated: vec![false; steps * agents],
            truncated: vec![false; steps * agents],
            final_values: vec![0.0; steps * agents],
            last_values: vec![0.0; agents],
            learner_agents: agents,
            learner_team_size: vec![0; agents],
            forward_rows: agents,
        }
    }
}

/// Worker -> main-thread reply. `Step`/`Reset`/`Debug` populate the step-shaped
/// buffers and leave `error` as `None`; `SetWeights` leaves the buffers empty and
/// uses `error` to signal ack (`None`) vs failure (`Some(msg)`). `Collect` leaves
/// the step-shaped buffers empty and populates `collect` on success (or `error`
/// on failure, same as `SetWeights`). One struct (rather than a response enum per
/// Cmd) keeps the worker's single `Sender<WorkerOut>` channel type unchanged.
struct WorkerOut {
    obs: Vec<f32>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    final_obs: Vec<f32>,
    debug_json: Option<String>,
    error: Option<String>,
    collect: Option<CollectOut>,
    /// `Cmd::RewardTerms` reply: this worker's arenas' counters, summed.
    terms: Option<[f64; crate::reward::N_TERMS]>,
}

impl WorkerOut {
    fn empty() -> Self {
        WorkerOut {
            obs: vec![],
            rewards: vec![],
            terminated: vec![],
            truncated: vec![],
            final_obs: vec![],
            debug_json: None,
            error: None,
            collect: None,
            terms: None,
        }
    }

    fn ack() -> Self {
        WorkerOut::empty()
    }

    fn err(msg: String) -> Self {
        WorkerOut { error: Some(msg), ..WorkerOut::empty() }
    }

    fn collect(out: CollectOut) -> Self {
        WorkerOut { collect: Some(out), ..WorkerOut::empty() }
    }
}

/// Parses a Python state_dict's raw arrays (`name -> (flat row-major data, shape)`)
/// into `PolicyWeights`. Expects PyTorch `nn.Sequential` trunk layout: `trunk.{i}.weight`
/// / `trunk.{i}.bias` for the Linear sublayers (ReLU occupies the odd indices, so `i`
/// is even: 0, 2, 4, ...), plus `policy_head.{weight,bias}` and `value_head.{weight,bias}`.
pub fn parse_state_dict(
    arrays: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<PolicyWeights, String> {
    fn take_layer(
        arrays: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        prefix: &str,
    ) -> Result<LayerWeights, String> {
        let (w, wshape) = arrays.get(&format!("{prefix}.weight"))
            .ok_or_else(|| format!("missing {prefix}.weight"))?;
        let (b, bshape) = arrays.get(&format!("{prefix}.bias"))
            .ok_or_else(|| format!("missing {prefix}.bias"))?;
        if wshape.len() != 2 || bshape.len() != 1 || bshape[0] != wshape[0] {
            return Err(format!("bad shapes for {prefix}: {wshape:?} / {bshape:?}"));
        }
        Ok(LayerWeights {
            w: w.clone(), b: b.clone(), out_dim: wshape[0], in_dim: wshape[1],
        })
    }

    // trunk.N.weight for even N (nn.Sequential interleaves ReLU at odd indices)
    let mut trunk_ids: Vec<usize> = arrays.keys()
        .filter_map(|k| k.strip_prefix("trunk.")?.strip_suffix(".weight")?.parse().ok())
        .collect();
    trunk_ids.sort_unstable();
    if trunk_ids.is_empty() {
        return Err("no trunk layers found".into());
    }
    if !trunk_ids.iter().enumerate().all(|(i, &id)| id == i * 2) {
        return Err(format!(
            "trunk layer indices must be 0,2,4,... (nn.Sequential with interleaved ReLU), got {trunk_ids:?}"
        ));
    }
    let trunk = trunk_ids.iter()
        .map(|i| take_layer(&arrays, &format!("trunk.{i}")))
        .collect::<Result<Vec<_>, _>>()?;
    let policy = take_layer(&arrays, "policy_head")?;
    let value = take_layer(&arrays, "value_head")?;

    // chain consistency: each trunk layer's in_dim must match the previous layer's out_dim
    for i in 1..trunk.len() {
        if trunk[i].in_dim != trunk[i - 1].out_dim {
            return Err(format!(
                "trunk layer {i} in_dim {} does not match layer {} out_dim {}",
                trunk[i].in_dim, i - 1, trunk[i - 1].out_dim
            ));
        }
    }
    if policy.in_dim != trunk.last().unwrap().out_dim
        || value.in_dim != trunk.last().unwrap().out_dim
        || value.out_dim != 1
    {
        return Err("head shapes do not match trunk output".into());
    }
    if trunk[0].in_dim != crate::obs::OBS_SIZE {
        return Err(format!(
            "trunk input dim {} != engine obs size {}", trunk[0].in_dim, crate::obs::OBS_SIZE
        ));
    }
    if policy.out_dim != crate::actions::TABLE_SIZE {
        return Err(format!(
            "policy output dim {} != engine action table size {}", policy.out_dim, crate::actions::TABLE_SIZE
        ));
    }

    // Reject any key not consumed above (trunk.N.{weight,bias} for discovered N,
    // policy_head.*, value_head.*) — catches stale/extra heads silently ignored otherwise.
    let mut consumed: std::collections::HashSet<String> = trunk_ids.iter()
        .flat_map(|i| [format!("trunk.{i}.weight"), format!("trunk.{i}.bias")])
        .collect();
    consumed.insert("policy_head.weight".into());
    consumed.insert("policy_head.bias".into());
    consumed.insert("value_head.weight".into());
    consumed.insert("value_head.bias".into());
    if let Some(k) = arrays.keys().find(|k| !consumed.contains(k.as_str())) {
        return Err(format!("unexpected state_dict key: {k}"));
    }

    Ok(PolicyWeights { trunk, policy, value })
}

/// One worker's v1 (entity-obs) rollout: the exact structure of the v0
/// `Cmd::Collect` arm — per-round {build obs for all agents, ONE batched
/// learner forward + one per used opponent slot, sample EVERY agent from its
/// arena's own rng in arena-major blue-then-orange order, step arenas,
/// final-value forward for done learner rows} then a post-rollout bootstrap
/// forward — with the flat 94-float obs replaced by ents/mask/query/prev
/// tensors and `MlpPolicy` by `EntityPolicy`. Kept as a separate function
/// (rather than branching inside the v0 loop) so the v0 path stays literally
/// untouched; the sampling order/rng-stream discipline is identical, so the
/// fixed-(seed, num_arenas, num_threads, schema) determinism contract holds
/// the same way. `emit_v0_obs` additionally records the legacy 94-float obs
/// per learner row for the kickstart teacher (T7).
#[allow(clippy::too_many_arguments)]
fn collect_v1_worker(
    steps: usize,
    my_assignment: &[i32],
    arenas: &mut [EpisodeArena],
    arena_sizes: &[(usize, usize)],
    a_to_arena: &[usize],
    max_arena_agents: usize,
    rngs: &mut [Pcg32],
    pol: &EntityPolicy,
    opponents: &[EntityPolicy],
    foreign: &mut [crate::foreign::ForeignPolicy],
    // Index of this worker's first arena in the GLOBAL arena list; used to build
    // per-arena keys for foreign state (car ids restart at 1 in every arena).
    global_base: usize,
    emit_v0_obs: bool,
) -> Result<CollectOut, String> {
    let agents = a_to_arena.len();
    // per-agent buffer widths
    let ek = MAX_ENT * ENT_FEAT;
    let (mk, qk, pk) = (MAX_ENT, Q_FEAT, PREV_ACTIONS);
    // Read the width off the POLICY, not a compiled constant: v1 has two legal
    // action tables (92-row v1.1, 104-row v1-air) and this value strides every
    // per-agent slice of `logits` below. A constant here would mis-slice every
    // agent after the first the moment a v1-air checkpoint is loaded.
    let action_count = pol.table_size();

    // Learner/opponent index maps — same construction as the v0 arm.
    let mut learner_idx: Vec<usize> = Vec::with_capacity(agents);
    let mut opp_idx: Vec<Vec<usize>> = vec![Vec::new(); opponents.len()];
    let mut learner_col: Vec<Option<usize>> = vec![None; agents];
    // Every agent whose action comes from OUR policy: `learner_idx` plus the
    // "mirror" cars of a PARTIAL foreign team (E4) -- orange cars in a foreign
    // arena that the ported bot does not drive. Mirrors are forwarded (so they
    // play, rather than acting uniformly at random) but are NOT learner rows:
    // their experience is discarded, which keeps the equal-rows arithmetic in
    // configs/train_v9_fromscratch.toml exact and keeps every learner row off a
    // team that contains a bot.
    //
    // With no partial foreign teams this is EQUAL to learner_idx, element for
    // element, so the batched forward's composition -- and therefore its float
    // rounding -- is unchanged from the pre-2026-07-26 build.
    let mut fwd_idx: Vec<usize> = Vec::with_capacity(agents);
    // Per learner ROW, the team size of the arena it came from (E5). Built in
    // THE SAME loop as learner_idx so the two orderings cannot drift: the
    // trainer standardises advantages per team size off this label, and a silent
    // mis-label would mis-scale the policy gradient undetectably.
    let mut learner_team_size: Vec<i8> = Vec::with_capacity(agents);
    // Arenas whose ORANGE cars are (partly) driven by a foreign (ported) bot:
    // (local arena index, foreign slot, first agent index of the arena).
    let mut foreign_arenas: Vec<(usize, usize, usize)> = Vec::new();
    {
        let mut a_off = 0usize;
        for (li, &(b, o)) in arena_sizes.iter().enumerate() {
            let k = my_assignment[li];
            if k <= -2 {
                // Foreign opponent: blue are learner rows, orange are driven by
                // `foreign[slot]` via a controls override at step time -- they are
                // neither learner rows nor native-opponent rows.
                let fslot = (-k - 2) as usize;
                // EVERY ported bot drives a team arena as of 2026-07-26. Nexto
                // and Necto are EARL attention models that size their obs from
                // `state.cars.len()` (obs_nexto::n_entities /
                // obs_necto::n_entities) with an all-false mask at a full
                // complement, so they see the whole arena unchanged; measured
                // at 9000 steps/arena vs a random opponent, goals/600s 67/71/67
                // (nexto) and 62/61/65 (necto) at 1v1/2v2/3v3 -- team size costs
                // them nothing.
                //
                // Immortal and Element consume AdvancedObs-107, a FIXED width
                // that fits exactly one other car, and until 2026-07-26 they
                // were REFUSED here because `build_advanced_obs` ran off the end
                // of the buffer at 2v2. That refusal treated a property of the
                // EXTRACTED ARTIFACT as a property of the bot. Upstream
                // RLMarlbot shares the limitation -- its advanced_obs.py loops
                // over all other cars with no padding and would emit 169 floats
                // into a 107-wide net -- but nothing stops US truncating: they
                // now see the opponent nearest the ball and nothing else
                // (foreign.rs Immortal/Element arms ->
                // obs_advanced::build_advanced_obs_one_other, which is
                // byte-identical to the old path at 1v1, so every existing 1v1
                // bench row stays comparable).
                //
                // Measured over 4000 steps/row x 3 seeds vs a random opponent, a
                // truncated element/immortal at 2v2 and 3v3 is indistinguishable
                // from the same bot at 1v1: element goals/600s 58+-5 (3v3) vs
                // 61+-5 (1v1), ball-facing 0.570 vs 0.555, approach 0.579 vs
                // 0.531, no spin, no idling, zero insane physics states. Against
                // nexto they track a NON-truncated necto control cell for cell as
                // team size grows, so the collapse with team size is the cost of
                // playing nexto rather than the cost of truncation.
                //
                // WHY WE WANT THEM HERE: they are the two easiest bots we have
                // (at a shared p4 vs ck_001171502080: immortal 0.896, element
                // 0.677, nexto 0.146) and were excluded from exactly the 2v2/3v3
                // blocks where the net is weakest and the lexicographic rung
                // controller has nowhere easier to go. They are also ~5x cheaper
                // per car-forward than nexto at 3v3 (0.23-0.26 ms vs 1.29 ms).
                //
                // TRIPWIRE: the bot cannot see 1 (2v2) or 3 (3v3) of our cars, so
                // "occupy the visible slot with car A, walk car B in" is a free
                // goal that exists against no real opponent. near-ball selection
                // bounds it (the car contesting the ball IS the visible one), as
                // does the [0.35, 0.65] auto-curriculum band. If win share vs an
                // element/immortal team slot ratchets to the easy pin and STAYS
                // there while the nexto/necto slots do not, that is the exploit
                // -- pull the slots. See docs/foreign-opponents.md.
                // BLUE only are learner rows; the bot drives the first
                // `cars_in_arena` orange cars and OUR policy mirrors the rest.
                let fcars = foreign.get(fslot).map(|f| f.cars_in_arena(o)).unwrap_or(o);
                for i in 0..b {
                    learner_idx.push(a_off + i);
                    learner_team_size.push(b.max(o) as i8);
                    fwd_idx.push(a_off + i);
                }
                for i in (b + fcars)..(b + o) {
                    fwd_idx.push(a_off + i); // mirror: forwarded, not learned from
                }
                foreign_arenas.push((li, fslot, a_off));
            } else if k < 0 {
                for i in 0..(b + o) {
                    learner_idx.push(a_off + i);
                    learner_team_size.push(b.max(o) as i8);
                    fwd_idx.push(a_off + i);
                }
            } else {
                let slot = k as usize;
                for i in 0..b {
                    learner_idx.push(a_off + i);
                    learner_team_size.push(b.max(o) as i8);
                    fwd_idx.push(a_off + i);
                }
                for i in b..(b + o) {
                    opp_idx[slot].push(a_off + i);
                }
            }
            a_off += b + o;
        }
        for (col, &a) in learner_idx.iter().enumerate() {
            learner_col[a] = Some(col);
        }
    }
    let n_learner = learner_idx.len();
    let n_fwd = fwd_idx.len();
    debug_assert_eq!(learner_team_size.len(), n_learner);

    let mut out = CollectOut::zeros_v1(steps, n_learner, emit_v0_obs);
    out.learner_team_size = learner_team_size;
    out.forward_rows = n_fwd;

    // Full-agent-width per-round obs scratch (current + terminal-final).
    let mut ents_buf = vec![0f32; agents * ek];
    let mut mask_buf = vec![false; agents * mk];
    let mut query_buf = vec![0f32; agents * qk];
    let mut prev_buf = vec![0i64; agents * pk];
    let mut v0_buf = vec![0f32; if emit_v0_obs { agents * OBS_SIZE } else { 0 }];
    let mut fin_ents = vec![0f32; agents * ek];
    let mut fin_mask = vec![false; agents * mk];
    let mut fin_query = vec![0f32; agents * qk];
    let mut fin_prev = vec![0i64; agents * pk];

    let mut rew_buf = vec![0f32; max_arena_agents];
    let mut flag_buf = vec![StepFlags::default(); max_arena_agents];
    let mut logits_all = vec![0f32; agents * action_count];
    let mut acts = vec![0i64; agents];

    // Batched-forward gather buffers (our policy + per opponent slot). Sized by
    // `n_fwd` (learner rows + partial-team mirrors), which equals `n_learner`
    // whenever no arena runs a partial foreign team.
    let mut l_ents = vec![0f32; n_fwd * ek];
    let mut l_mask = vec![false; n_fwd * mk];
    let mut l_query = vec![0f32; n_fwd * qk];
    let mut l_prev = vec![0i64; n_fwd * pk];
    let mut o_ents: Vec<Vec<f32>> = opp_idx.iter().map(|ix| vec![0f32; ix.len() * ek]).collect();
    let mut o_mask: Vec<Vec<bool>> = opp_idx.iter().map(|ix| vec![false; ix.len() * mk]).collect();
    let mut o_query: Vec<Vec<f32>> = opp_idx.iter().map(|ix| vec![0f32; ix.len() * qk]).collect();
    let mut o_prev: Vec<Vec<i64>> = opp_idx.iter().map(|ix| vec![0i64; ix.len() * pk]).collect();

    // gather one agent's rows from the full-width buffers into row j of a batch
    macro_rules! gather {
        ($j:expr, $a:expr, $de:expr, $dm:expr, $dq:expr, $dp:expr,
         $se:expr, $sm:expr, $sq:expr, $sp:expr) => {{
            let (j, a) = ($j, $a);
            $de[j * ek..(j + 1) * ek].copy_from_slice(&$se[a * ek..(a + 1) * ek]);
            $dm[j * mk..(j + 1) * mk].copy_from_slice(&$sm[a * mk..(a + 1) * mk]);
            $dq[j * qk..(j + 1) * qk].copy_from_slice(&$sq[a * qk..(a + 1) * qk]);
            $dp[j * pk..(j + 1) * pk].copy_from_slice(&$sp[a * pk..(a + 1) * pk]);
        }};
    }

    for t in 0..steps {
        // 1. v1 obs for all arenas (every agent acts this round); plus the
        // legacy obs when the kickstart teacher needs it.
        let mut off = 0usize;
        for ar in arenas.iter_mut() {
            let n = ar.num_agents();
            ar.write_obs_v1(
                &mut ents_buf[off * ek..(off + n) * ek],
                &mut mask_buf[off * mk..(off + n) * mk],
                &mut query_buf[off * qk..(off + n) * qk],
                &mut prev_buf[off * pk..(off + n) * pk],
            );
            if emit_v0_obs {
                ar.write_obs(&mut v0_buf[off * OBS_SIZE..(off + n) * OBS_SIZE]);
            }
            off += n;
        }

        // 2. record obs for LEARNER rows only.
        for (j, &a) in learner_idx.iter().enumerate() {
            let r = t * n_learner + j;
            out.ents[r * ek..(r + 1) * ek].copy_from_slice(&ents_buf[a * ek..(a + 1) * ek]);
            out.mask[r * mk..(r + 1) * mk].copy_from_slice(&mask_buf[a * mk..(a + 1) * mk]);
            out.query[r * qk..(r + 1) * qk].copy_from_slice(&query_buf[a * qk..(a + 1) * qk]);
            out.prev[r * pk..(r + 1) * pk].copy_from_slice(&prev_buf[a * pk..(a + 1) * pk]);
            if emit_v0_obs {
                out.obs_v0[r * OBS_SIZE..(r + 1) * OBS_SIZE]
                    .copy_from_slice(&v0_buf[a * OBS_SIZE..(a + 1) * OBS_SIZE]);
            }
        }

        // 3. batched forwards: one learner batch + one per used opponent slot,
        // logits scattered into full agent width for uniform sampling below.
        if n_fwd > 0 {
            for (j, &a) in fwd_idx.iter().enumerate() {
                gather!(j, a, l_ents, l_mask, l_query, l_prev, ents_buf, mask_buf, query_buf, prev_buf);
            }
            let (l_logits, l_values) = pol.forward(&l_ents, &l_mask, &l_query, &l_prev, n_fwd)?;
            // Scatter logits for EVERY forwarded agent, but record values only
            // for learner rows. When there are no mirrors, `fwd_idx` IS
            // `learner_idx` and this writes `out.values` in the same order the
            // old bulk copy_from_slice did.
            for (j, &a) in fwd_idx.iter().enumerate() {
                logits_all[a * action_count..(a + 1) * action_count]
                    .copy_from_slice(&l_logits[j * action_count..(j + 1) * action_count]);
                if let Some(col) = learner_col[a] {
                    out.values[t * n_learner + col] = l_values[j];
                }
            }
        }
        for (slot, idxs) in opp_idx.iter().enumerate() {
            if idxs.is_empty() {
                continue;
            }
            for (j, &a) in idxs.iter().enumerate() {
                gather!(j, a, o_ents[slot], o_mask[slot], o_query[slot], o_prev[slot],
                        ents_buf, mask_buf, query_buf, prev_buf);
            }
            let (o_logits, _) = opponents[slot]
                .forward(&o_ents[slot], &o_mask[slot], &o_query[slot], &o_prev[slot], idxs.len())?;
            for (j, &a) in idxs.iter().enumerate() {
                logits_all[a * action_count..(a + 1) * action_count]
                    .copy_from_slice(&o_logits[j * action_count..(j + 1) * action_count]);
            }
        }

        // 4. sample EVERY agent — same order and per-arena rng stream as v0.
        for a in 0..agents {
            let row = &logits_all[a * action_count..(a + 1) * action_count];
            let (idx, lp) = sample_categorical(row, &mut rngs[a_to_arena[a]]);
            acts[a] = idx as i64;
            if let Some(col) = learner_col[a] {
                out.actions[t * n_learner + col] = idx as i64;
                out.logprobs[t * n_learner + col] = lp;
            }
        }

        // 5. step arenas (all agents), record learner rewards/flags.
        let mut aoff = 0usize;
        let mut done: Vec<usize> = Vec::new();
        for (li, ar) in arenas.iter_mut().enumerate() {
            let n = ar.num_agents();
            // Foreign-driven arena: build this arena's orange cars' controls from
            // the ported bot (its own obs/net/table) and queue them as a one-shot
            // override, so step_impl uses them instead of our action table.
            let mut foreign_ids: Vec<u64> = Vec::new();
            if let Some(&(_, fslot, _)) = foreign_arenas.iter().find(|(l, _, _)| *l == li) {
                let bc = ar.blue_count();
                // PARTIAL foreign team (E4): the bot drives the first `k` orange
                // cars, the rest keep the action our policy already sampled for
                // them. `k` defaults to the whole orange side, so a run that
                // never sets foreign_cars queues exactly the same overrides as
                // before. Must match the `fwd_idx` split above -- both take the
                // FIRST k orange cars.
                let k = foreign[fslot].cars_in_arena(n - bc);
                let mut ov = Vec::with_capacity(k);
                for i in bc..(bc + k) {
                    let (cid, key, ctrl) =
                        ar.foreign_controls(i, &mut foreign[fslot], global_base + li);
                    foreign_ids.push(key);
                    ov.push((cid, ctrl));
                }
                ar.set_foreign_overrides(ov);
            }
            ar.step_v1(
                &acts[aoff..aoff + n],
                &mut rew_buf[..n],
                &mut flag_buf[..n],
                &mut fin_ents[aoff * ek..(aoff + n) * ek],
                &mut fin_mask[aoff * mk..(aoff + n) * mk],
                &mut fin_query[aoff * qk..(aoff + n) * qk],
                &mut fin_prev[aoff * pk..(aoff + n) * pk],
            );
            for i in 0..n {
                let a = aoff + i;
                if let Some(col) = learner_col[a] {
                    out.rewards[t * n_learner + col] = rew_buf[i];
                    out.terminated[t * n_learner + col] = flag_buf[i].terminated;
                    out.truncated[t * n_learner + col] = flag_buf[i].truncated;
                    if flag_buf[i].terminated || flag_buf[i].truncated {
                        done.push(a);
                    }
                }
            }
            // Episode boundary in a foreign arena: clear just THIS arena's cars'
            // stored previous action so the bot's obs restarts from zeros (a slot
            // may drive several arenas, so a blanket reset() would be wrong).
            if !foreign_ids.is_empty() && flag_buf[..n].iter().any(|f| f.terminated || f.truncated)
            {
                if let Some(&(_, fslot, _)) = foreign_arenas.iter().find(|(l, _, _)| *l == li) {
                    for key in &foreign_ids {
                        foreign[fslot].reset_car(*key);
                    }
                }
            }
            aoff += n;
        }

        // 6. final values for done rows (learner-only by construction) —
        // single batched forward over the final v1 obs.
        if !done.is_empty() {
            let nd = done.len();
            let mut d_ents = vec![0f32; nd * ek];
            let mut d_mask = vec![false; nd * mk];
            let mut d_query = vec![0f32; nd * qk];
            let mut d_prev = vec![0i64; nd * pk];
            for (j, &a) in done.iter().enumerate() {
                gather!(j, a, d_ents, d_mask, d_query, d_prev, fin_ents, fin_mask, fin_query, fin_prev);
            }
            let (_, fv) = pol.forward(&d_ents, &d_mask, &d_query, &d_prev, nd)?;
            for (j, &a) in done.iter().enumerate() {
                let col = learner_col[a].expect("done rows are learner rows by construction");
                out.final_values[t * n_learner + col] = fv[j];
            }
        }
    }

    // 7. bootstrap values of the post-rollout obs, learner rows only.
    let mut off = 0usize;
    for ar in arenas.iter_mut() {
        let n = ar.num_agents();
        ar.write_obs_v1(
            &mut ents_buf[off * ek..(off + n) * ek],
            &mut mask_buf[off * mk..(off + n) * mk],
            &mut query_buf[off * qk..(off + n) * qk],
            &mut prev_buf[off * pk..(off + n) * pk],
        );
        off += n;
    }
    if n_learner > 0 {
        // Learner rows ONLY here (not `fwd_idx`): `last_values` is the GAE
        // bootstrap for the rows that actually become training data, and a
        // mirror row has none.
        for (j, &a) in learner_idx.iter().enumerate() {
            gather!(j, a, l_ents, l_mask, l_query, l_prev, ents_buf, mask_buf, query_buf, prev_buf);
        }
        // The gather buffers are sized for `n_fwd` rows; slice them to the
        // `n_learner` rows actually filled above, or `forward` infers max_ent
        // from `mask.len() / b` and rejects the batch.
        let (_, lv) = pol.forward(
            &l_ents[..n_learner * ek],
            &l_mask[..n_learner * mk],
            &l_query[..n_learner * qk],
            &l_prev[..n_learner * pk],
            n_learner,
        )?;
        out.last_values.copy_from_slice(&lv);
    }
    Ok(out)
}

/// Split num_arenas into team sizes 1/2/3 proportional to `weights`
/// (largest-remainder method), ordered as a 1s block, 2s block, 3s block.
pub fn allocate_team_sizes(num_arenas: usize, weights: [f64; 3]) -> Vec<usize> {
    let total: f64 = weights.iter().sum();
    assert!(total > 0.0, "team size weights must sum > 0");
    let exact: Vec<f64> = weights.iter().map(|w| w / total * num_arenas as f64).collect();
    let mut counts: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
    let mut short = num_arenas - counts.iter().sum::<usize>();
    // hand out remainders by largest fractional part; ties -> smaller size first (stable)
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by(|&a, &b| {
        let fa = exact[a] - exact[a].floor();
        let fb = exact[b] - exact[b].floor();
        fb.partial_cmp(&fa).unwrap().then(a.cmp(&b))
    });
    for &i in &order {
        if short == 0 {
            break;
        }
        counts[i] += 1;
        short -= 1;
    }
    let mut out = Vec::with_capacity(num_arenas);
    for (i, &c) in counts.iter().enumerate() {
        out.extend(std::iter::repeat(i + 1).take(c));
    }
    out
}

struct Worker {
    tx: Sender<Cmd>,
    rx: Receiver<WorkerOut>,
    handle: Option<JoinHandle<()>>,
    num_agents: usize,
    num_arenas: usize,
}

pub struct MultiEngine {
    workers: Vec<Worker>,
    pub num_agents: usize,
    pub num_arenas: usize,
    pub obs_size: usize,
    pub action_count: usize,
    /// Obs family, derived from `schema.version` at construction (0 -> V0,
    /// 1 -> V1). Drives `set_weights`/`set_opponents` variant validation and
    /// the collect buffer layout.
    pub obs_mode: ObsMode,
    /// V1-only: also record the legacy 94-float obs per learner row into
    /// `CollectOut::obs_v0` (kickstart teacher input, T7).
    emit_v0_obs: bool,
    // Debug-forward policy copy — see `set_weights`/`debug_policy_forward` doc comments
    // for why this lives here instead of routing through a worker.
    debug_policy: Option<MlpPolicy>,
    // `sizes[arena] = (blue, orange)`, kept (in addition to being consumed per-worker
    // at construction) so `collect`'s entry validation can compute the exact
    // learner-row count for an assignment (self-play arenas contribute all agents,
    // opponent arenas contribute only their blue agents) without a worker round-trip.
    sizes: Vec<(usize, usize)>,
    // Number of currently-set opponent slots (`Cmd::SetOpponents` payload length);
    // 0 until `set_opponents` is first called. Bounds-checks `Collect`'s assignment.
    opponent_slots: usize,
    // Number of currently-set FOREIGN (ported-bot) opponent slots. Separate slot
    // space from `opponent_slots`; a Collect assignment addresses foreign slot `f`
    // as `-(f) - 2`. 0 until `set_foreign_opponents` is called.
    foreign_slots: usize,
}

impl MultiEngine {
    /// `sizes[arena] = (blue, orange)` cars-per-team for that arena, one entry per
    /// arena (so `sizes.len()` is the arena count). Uniform legacy construction is
    /// `vec![(blue, orange); num_arenas]`; mixed-team-size construction (Task 2) maps
    /// `engine::allocate_team_sizes` output `s` to `(s, s)` per arena. Kept as pairs
    /// (rather than a single size) so the asymmetric case (e.g. blue != orange, used
    /// by tests) keeps working uniformly with the mixed-size path.
    /// `emit_v0_obs` is meaningful only for schema v1 (record the legacy
    /// 94-float obs alongside the entity tensors, for the kickstart
    /// teacher); v0 callers pass `false` (lib.rs rejects `true` with a v0
    /// schema before constructing).
    pub fn new(
        sizes: Vec<(usize, usize)>,
        schema: Schema,
        reward_cfg: RewardConfig,
        seed: u32,
        num_threads: usize,
        curriculum: Option<CurriculumConfig>,
        emit_v0_obs: bool,
    ) -> Self {
        let obs_mode = if schema.version == 1 { ObsMode::V1 } else { ObsMode::V0 };
        // Which V1 action table the schema selected; moved into every worker
        // below so each arena decodes with the same table the policy emits.
        let act_table = schema.action_table_kind();
        let num_arenas = sizes.len();
        let threads = if num_threads == 0 {
            std::thread::available_parallelism().map(|n| n.get().saturating_sub(2).max(1)).unwrap_or(4)
        } else {
            num_threads
        }
        .min(num_arenas);

        // distribute arenas round-robin-contiguously over threads
        let mut workers = Vec::with_capacity(threads);
        let mut assigned = 0usize;
        for t in 0..threads {
            let count = (num_arenas - assigned) / (threads - t); // even split
            assigned += count;
            let (ctx, crx) = channel::<Cmd>();
            let (otx, orx) = channel::<WorkerOut>();
            let (sch, cfg, curr) = (schema.clone(), reward_cfg.clone(), curriculum.clone());
            // Seed by GLOBAL arena index: this makes each arena's own sim state
            // (kickoff RNG, per-arena action-sample RNG below) invariant to how
            // arenas are sharded across worker threads. That is NOT the same as
            // collect()'s end-to-end determinism contract, which is narrower —
            // see the fuller comment below on why num_threads still affects the
            // batched-forward float rounding.
            let global_base = assigned - count;
            let arena_sizes: Vec<(usize, usize)> = sizes[global_base..global_base + count].to_vec();
            let worker_agents: usize = arena_sizes.iter().map(|&(b, o)| b + o).sum();
            let handle = std::thread::spawn(move || {
                // arenas created inside the worker thread
                let mut arenas: Vec<EpisodeArena> = arena_sizes
                    .iter()
                    .enumerate()
                    .map(|(i, &(b, o))| {
                        EpisodeArena::new_full_with_table(b, o, sch.tick_skip, cfg.clone(),
                                          sch.normalization.clone(), seed.wrapping_add((global_base + i) as u32),
                                          curr.clone(), obs_mode, act_table)
                    })
                    .collect();
                // Per-arena agent counts (blue+orange, may vary across arenas now)
                // and the flattened agent->arena lookup used by the Collect sampling
                // loop below (replaces the old uniform `a / per_agent` division).
                let arena_agent_counts: Vec<usize> = arenas.iter().map(|ar| ar.num_agents()).collect();
                let agents: usize = arena_agent_counts.iter().sum();
                debug_assert_eq!(agents, worker_agents);
                let mut a_to_arena = Vec::with_capacity(agents);
                for (ai, &c) in arena_agent_counts.iter().enumerate() {
                    a_to_arena.extend(std::iter::repeat(ai).take(c));
                }
                // Scratch-buffer size for the largest single arena on this worker
                // (buffers below are reused across arenas within a round, sliced to
                // each arena's own agent count) — replaces the old uniform `per_agent`.
                let max_arena_agents = arena_agent_counts.iter().copied().max().unwrap_or(0);
                // One Pcg32 per ARENA (not per agent), seeded by GLOBAL arena index —
                // same key as the arena's own kickoff seed above. Each arena's agents
                // (blue-then-orange, matching `EpisodeArena::car_ids` order) draw from
                // that arena's rng only; arenas never share an rng, so the RNG stream
                // consumed per arena is fixed regardless of thread layout. Note this
                // does NOT make collect() thread-count invariant end to end — the
                // batched forward's float rounding varies with worker batch size (see
                // the Cmd::Collect arm); the determinism contract is fixed
                // (seed, num_arenas, num_threads) only.
                let mut rngs: Vec<Pcg32> = (0..count)
                    .map(|i| Pcg32::new((seed as u64) * 1_000_000 + (global_base + i) as u64))
                    .collect();
                // Holds the worker's own MlpPolicy, set via Cmd::SetWeights and read by
                // Cmd::Collect for on-worker rollout. `debug_policy_forward` (Task 3)
                // does not route through here — see MultiEngine::debug_policy_forward.
                let mut policy: Option<MlpPolicy> = None;
                // V1 twin of `policy` — exactly one of the two is ever Some,
                // decided by the NetWeights variant (which MultiEngine
                // validated against obs_mode before broadcasting).
                let mut policy_v1: Option<EntityPolicy> = None;
                // Opponent-policy slots (indexed by the `k >= 0` values in a
                // Collect assignment), rebuilt wholesale on every SetOpponents.
                // Empty until set_opponents is called — legacy (all-self-play)
                // collects never index into this.
                let mut opponents: Vec<MlpPolicy> = Vec::new();
                // V1 twin of `opponents` (same slot indexing).
                let mut opponents_v1: Vec<EntityPolicy> = Vec::new();
                // Foreign (ported-bot) opponent slots — a SEPARATE slot space,
                // addressed by `k <= -2` in a Collect assignment. Empty until
                // set_foreign_opponents is called.
                let mut opponents_foreign: Vec<crate::foreign::ForeignPolicy> = Vec::new();
                while let Ok(cmd) = crx.recv() {
                    match cmd {
                        Cmd::Shutdown => break,
                        Cmd::Reset => {
                            let mut out = WorkerOut {
                                obs: vec![0.0; agents * OBS_SIZE],
                                rewards: vec![0.0; agents],
                                terminated: vec![false; agents],
                                truncated: vec![false; agents],
                                final_obs: vec![0.0; agents * OBS_SIZE],
                                debug_json: None,
                                error: None,
                                collect: None,
                                terms: None,
                            };
                            let mut off = 0;
                            for ar in arenas.iter_mut() {
                                let n = ar.num_agents() * OBS_SIZE;
                                ar.write_obs(&mut out.obs[off..off + n]);
                                off += n;
                            }
                            let _ = otx.send(out);
                        }
                        Cmd::Step(acts) => {
                            let mut out = WorkerOut {
                                obs: vec![0.0; agents * OBS_SIZE],
                                rewards: vec![0.0; agents],
                                terminated: vec![false; agents],
                                truncated: vec![false; agents],
                                final_obs: vec![0.0; agents * OBS_SIZE],
                                debug_json: None,
                                error: None,
                                collect: None,
                                terms: None,
                            };
                            let mut a_off = 0;
                            let mut flags = vec![StepFlags::default(); max_arena_agents];
                            for ar in arenas.iter_mut() {
                                let n = ar.num_agents();
                                ar.step(
                                    &acts[a_off..a_off + n],
                                    &mut out.rewards[a_off..a_off + n],
                                    &mut flags[..n],
                                    &mut out.final_obs[a_off * OBS_SIZE..(a_off + n) * OBS_SIZE],
                                );
                                for (i, f) in flags[..n].iter().enumerate() {
                                    out.terminated[a_off + i] = f.terminated;
                                    out.truncated[a_off + i] = f.truncated;
                                }
                                ar.write_obs(&mut out.obs[a_off * OBS_SIZE..(a_off + n) * OBS_SIZE]);
                                a_off += n;
                            }
                            let _ = otx.send(out);
                        }
                        Cmd::RewardTerms => {
                            let mut sums = [0.0f64; crate::reward::N_TERMS];
                            for ar in arenas.iter_mut() {
                                for (s, v) in sums.iter_mut().zip(ar.take_reward_terms()) {
                                    *s += v;
                                }
                            }
                            let _ = otx.send(WorkerOut { terms: Some(sums), ..WorkerOut::empty() });
                        }
                        Cmd::Debug { local_idx } => {
                            let ar = &mut arenas[local_idx];
                            let n = ar.num_agents();
                            let mut out = WorkerOut {
                                obs: vec![0.0; n * OBS_SIZE],
                                rewards: vec![],
                                terminated: vec![],
                                truncated: vec![],
                                final_obs: vec![],
                                debug_json: None,
                                error: None,
                                collect: None,
                                terms: None,
                            };
                            ar.write_obs(&mut out.obs);
                            out.debug_json = Some(ar.debug_state_json());
                            let _ = otx.send(out);
                        }
                        Cmd::SetWeights(w) => {
                            let built = match &*w {
                                NetWeights::V0(pw) => MlpPolicy::new(pw).map(|p| {
                                    policy = Some(p);
                                    policy_v1 = None;
                                }),
                                NetWeights::V1 { raw, heads } => {
                                    EntityPolicy::new(raw, *heads).map(|p| {
                                        policy_v1 = Some(p);
                                        policy = None;
                                    })
                                }
                                // MultiEngine::set_weights rejects this before it
                                // ever reaches a worker; kept for exhaustiveness.
                                NetWeights::Foreign { .. } => {
                                    Err("a foreign bot cannot be the learner policy".to_string())
                                }
                            };
                            match built {
                                Ok(()) => { let _ = otx.send(WorkerOut::ack()); }
                                Err(e) => { let _ = otx.send(WorkerOut::err(e)); }
                            }
                        }
                        Cmd::SetOpponents(ws) => {
                            let mut built_v0 = Vec::new();
                            let mut built_v1 = Vec::new();
                            let mut build_err: Option<String> = None;
                            for w in ws.iter() {
                                let r = match w {
                                    NetWeights::V0(pw) => MlpPolicy::new(pw).map(|p| built_v0.push(p)),
                                    NetWeights::V1 { raw, heads } => {
                                        EntityPolicy::new(raw, *heads).map(|p| built_v1.push(p))
                                    }
                                    // Rejected by MultiEngine::set_opponents; foreign
                                    // bots use the separate SetForeignOpponents slot space.
                                    NetWeights::Foreign { .. } => Err(
                                        "foreign dict in set_opponents; use set_foreign_opponents"
                                            .to_string(),
                                    ),
                                };
                                if let Err(e) = r {
                                    build_err = Some(e);
                                    break;
                                }
                            }
                            match build_err {
                                Some(e) => { let _ = otx.send(WorkerOut::err(e)); }
                                None => {
                                    // MultiEngine::set_opponents validated the
                                    // variants match obs_mode, so exactly one
                                    // of these carries the slots.
                                    opponents = built_v0;
                                    opponents_v1 = built_v1;
                                    let _ = otx.send(WorkerOut::ack());
                                }
                            }
                        }
                        Cmd::SetForeignOpponents(ws) => {
                            let mut built = Vec::new();
                            let mut build_err: Option<String> = None;
                            for w in ws.iter() {
                                match w {
                                    NetWeights::Foreign { raw, kind, period, cars } => {
                                        match crate::foreign::ForeignPolicy::new(raw, *kind) {
                                            Ok(mut p) => {
                                                p.set_decision_period(*period);
                                                p.set_foreign_cars(*cars);
                                                built.push(p)
                                            }
                                            Err(e) => {
                                                build_err = Some(e);
                                                break;
                                            }
                                        }
                                    }
                                    _ => {
                                        build_err =
                                            Some("set_foreign_opponents got a non-foreign dict".into());
                                        break;
                                    }
                                }
                            }
                            match build_err {
                                Some(e) => {
                                    let _ = otx.send(WorkerOut::err(e));
                                }
                                None => {
                                    opponents_foreign = built;
                                    let _ = otx.send(WorkerOut::ack());
                                }
                            }
                        }
                        Cmd::Collect { steps, assignment } => {
                            // V1 (entity) rollout lives in its own function so
                            // the v0 loop below stays literally untouched
                            // (byte-identity gate). Same worker-batched
                            // forward + per-arena rng discipline.
                            if obs_mode == ObsMode::V1 {
                                let Some(pol) = policy_v1.as_ref() else {
                                    let _ = otx.send(WorkerOut::err("collect before set_weights".into()));
                                    continue;
                                };
                                let my_assignment = &assignment[global_base..global_base + count];
                                let msg = match collect_v1_worker(
                                    steps, my_assignment, &mut arenas, &arena_sizes,
                                    &a_to_arena, max_arena_agents, &mut rngs, pol,
                                    &opponents_v1, &mut opponents_foreign, global_base,
                                    emit_v0_obs,
                                ) {
                                    Ok(out) => WorkerOut::collect(out),
                                    Err(e) => WorkerOut::err(e),
                                };
                                let _ = otx.send(msg);
                                continue;
                            }
                            let Some(pol) = policy.as_ref() else {
                                let _ = otx.send(WorkerOut::err("collect before set_weights".into()));
                                continue;
                            };
                            let d = OBS_SIZE;
                            let action_count = actions::TABLE_SIZE;

                            // This worker's slice of the global (length num_arenas)
                            // assignment — arena `li` here is global arena
                            // `global_base + li`. MultiEngine::collect validates the
                            // whole vector (length, slot range, learner_count > 0)
                            // before sending any Cmd, so every value here is
                            // already known-good.
                            let my_assignment = &assignment[global_base..global_base + count];

                            // Build index lists ONCE per collect (not per round):
                            // learner_idx is every agent of a self-play arena
                            // (k == -1) plus the BLUE agents of an opponent arena
                            // (k >= 0, orange driven by opponents[k]); opp_idx[slot]
                            // is the ORANGE agents of arenas assigned to that slot.
                            // Both follow the existing arena-major, blue-then-orange
                            // agent order — for the legacy all-(-1) assignment this
                            // makes learner_idx == 0..agents and learner_col the
                            // identity map, so that path is byte-identical to the
                            // pre-league full-width layout below.
                            let mut learner_idx: Vec<usize> = Vec::with_capacity(agents);
                            let mut opp_idx: Vec<Vec<usize>> = vec![Vec::new(); opponents.len()];
                            let mut learner_col: Vec<Option<usize>> = vec![None; agents];
                            {
                                let mut a_off = 0usize;
                                for (li, &(b, o)) in arena_sizes.iter().enumerate() {
                                    let k = my_assignment[li];
                                    if k < 0 {
                                        for i in 0..(b + o) {
                                            learner_idx.push(a_off + i);
                                        }
                                    } else {
                                        let slot = k as usize;
                                        for i in 0..b {
                                            learner_idx.push(a_off + i);
                                        }
                                        for i in b..(b + o) {
                                            opp_idx[slot].push(a_off + i);
                                        }
                                    }
                                    a_off += b + o;
                                }
                                for (col, &a) in learner_idx.iter().enumerate() {
                                    learner_col[a] = Some(col);
                                }
                            }
                            let n_learner = learner_idx.len();

                            let mut out = CollectOut::zeros(steps, n_learner, d);
                            let mut obs_buf = vec![0f32; agents * d];
                            let mut rew_buf = vec![0f32; max_arena_agents];
                            let mut flag_buf = vec![StepFlags::default(); max_arena_agents];
                            let mut fin_buf = vec![0f32; agents * d];
                            // Full-agent-width scratch that steps 2/3 scatter every
                            // agent's logits into (learner forward + per-slot
                            // opponent forwards both write here) so step 3 can
                            // sample every agent uniformly.
                            let mut logits_all = vec![0f32; agents * action_count];
                            let mut learner_obs = vec![0f32; n_learner * d];
                            let mut opp_obs_bufs: Vec<Vec<f32>> =
                                opp_idx.iter().map(|idxs| vec![0f32; idxs.len() * d]).collect();
                            let mut acts = vec![0i64; agents];

                            let mut worker_err: Option<String> = None;
                            'rounds: for t in 0..steps {
                                // 1. obs for all my arenas (every agent needs an obs
                                // this round, learner or opponent-driven, to act).
                                let mut off = 0;
                                for ar in arenas.iter_mut() {
                                    let n = ar.num_agents() * d;
                                    ar.write_obs(&mut obs_buf[off..off + n]);
                                    off += n;
                                }
                                // Record obs for LEARNER rows only.
                                for (j, &a) in learner_idx.iter().enumerate() {
                                    out.obs[t * n_learner * d + j * d..t * n_learner * d + (j + 1) * d]
                                        .copy_from_slice(&obs_buf[a * d..(a + 1) * d]);
                                }

                                // 2. Forwards: ONE batched learner forward over
                                // learner_idx rows, plus one batched forward per
                                // USED opponent slot over that slot's opp_idx rows —
                                // no per-agent forwards. Results are scattered into
                                // logits_all (full agent width). For the legacy
                                // all-(-1) assignment this is exactly the old single
                                // whole-worker forward (learner_idx == 0..agents),
                                // so the batch-size-dependent gemm rounding noted
                                // below is unchanged for that path. NOTE: this makes
                                // float outputs (values/logits) depend on the
                                // worker's batch size, which depends on how arenas
                                // are split across threads — candle's CPU gemm
                                // rounds the *same* input row differently at
                                // different batch sizes (~1e-7 in the logits,
                                // verified empirically). Cross-thread-count
                                // determinism is therefore NOT provided: those
                                // ~1e-7 logit differences occasionally flip which
                                // CDF bucket a sample lands in, actions diverge,
                                // and trajectories separate entirely — so no useful
                                // cross-thread-count guarantee exists at any
                                // tolerance. The determinism contract is exact
                                // reproducibility for a fixed (seed, num_arenas,
                                // num_threads) config only. A per-arena-forward
                                // variant that WAS thread-count exact was tried and
                                // reverted: batch=per_agent gemm calls are
                                // overhead-dominated (18k env-steps/s vs the Python
                                // path's ~45k at 96 arenas).
                                if n_learner > 0 {
                                    for (j, &a) in learner_idx.iter().enumerate() {
                                        learner_obs[j * d..(j + 1) * d].copy_from_slice(&obs_buf[a * d..(a + 1) * d]);
                                    }
                                    let (l_logits, l_values) = match pol.forward(&learner_obs, n_learner, d) {
                                        Ok(x) => x,
                                        Err(e) => { worker_err = Some(e); break 'rounds; }
                                    };
                                    for (j, &a) in learner_idx.iter().enumerate() {
                                        logits_all[a * action_count..(a + 1) * action_count]
                                            .copy_from_slice(&l_logits[j * action_count..(j + 1) * action_count]);
                                    }
                                    out.values[t * n_learner..(t + 1) * n_learner].copy_from_slice(&l_values);
                                }
                                for (slot, idxs) in opp_idx.iter().enumerate() {
                                    if idxs.is_empty() {
                                        continue;
                                    }
                                    {
                                        let buf = &mut opp_obs_bufs[slot];
                                        for (j, &a) in idxs.iter().enumerate() {
                                            buf[j * d..(j + 1) * d].copy_from_slice(&obs_buf[a * d..(a + 1) * d]);
                                        }
                                    }
                                    let (o_logits, _) = match opponents[slot].forward(&opp_obs_bufs[slot], idxs.len(), d) {
                                        Ok(x) => x,
                                        Err(e) => { worker_err = Some(e); break 'rounds; }
                                    };
                                    for (j, &a) in idxs.iter().enumerate() {
                                        logits_all[a * action_count..(a + 1) * action_count]
                                            .copy_from_slice(&o_logits[j * action_count..(j + 1) * action_count]);
                                    }
                                }

                                // 3. sample for EVERY agent — agent a belongs to arena
                                // a_to_arena[a]; each arena's agents draw from that
                                // arena's own rng in blue-then-orange order, never
                                // shared across arenas. This runs unfiltered
                                // (learner AND opponent rows alike, same order as
                                // always) — only recording below is filtered — which
                                // is what keeps the rng stream, and therefore
                                // determinism and the byte-identity gate, intact.
                                for a in 0..agents {
                                    let row = &logits_all[a * action_count..(a + 1) * action_count];
                                    let (idx, lp) = sample_categorical(row, &mut rngs[a_to_arena[a]]);
                                    acts[a] = idx as i64;
                                    if let Some(col) = learner_col[a] {
                                        out.actions[t * n_learner + col] = idx as i64;
                                        out.logprobs[t * n_learner + col] = lp;
                                    }
                                }

                                // 4. step arenas (every agent, unfiltered — physics
                                // doesn't know about learner/opponent), record
                                // rewards/flags for LEARNER rows only.
                                let mut aoff = 0;
                                let mut done: Vec<usize> = Vec::new();
                                for ar in arenas.iter_mut() {
                                    let n = ar.num_agents();
                                    ar.step(
                                        &acts[aoff..aoff + n],
                                        &mut rew_buf[..n],
                                        &mut flag_buf[..n],
                                        &mut fin_buf[aoff * d..(aoff + n) * d],
                                    );
                                    for i in 0..n {
                                        let a = aoff + i;
                                        if let Some(col) = learner_col[a] {
                                            out.rewards[t * n_learner + col] = rew_buf[i];
                                            out.terminated[t * n_learner + col] = flag_buf[i].terminated;
                                            out.truncated[t * n_learner + col] = flag_buf[i].truncated;
                                            if flag_buf[i].terminated || flag_buf[i].truncated {
                                                done.push(a);
                                            }
                                        }
                                    }
                                    aoff += n;
                                }

                                // 5. final values for done rows — LEARNER rows only
                                // (single batched forward over just the done subset,
                                // worker-wide; `done` above is already learner-only
                                // by construction, so opponent rows never reach the
                                // learner policy here).
                                if !done.is_empty() {
                                    let mut fobs = vec![0f32; done.len() * d];
                                    for (j, &a) in done.iter().enumerate() {
                                        fobs[j * d..(j + 1) * d].copy_from_slice(&fin_buf[a * d..(a + 1) * d]);
                                    }
                                    match pol.forward(&fobs, done.len(), d) {
                                        Ok((_, fv)) => {
                                            for (j, &a) in done.iter().enumerate() {
                                                let col = learner_col[a]
                                                    .expect("done rows are learner rows by construction");
                                                out.final_values[t * n_learner + col] = fv[j];
                                            }
                                        }
                                        Err(e) => { worker_err = Some(e); break 'rounds; }
                                    }
                                }
                            }

                            if let Some(e) = worker_err {
                                let _ = otx.send(WorkerOut::err(e));
                                continue;
                            }

                            // 6. bootstrap values of the post-rollout obs, LEARNER
                            // rows only (one batched forward).
                            let mut off = 0;
                            for ar in arenas.iter_mut() {
                                let n = ar.num_agents() * d;
                                ar.write_obs(&mut obs_buf[off..off + n]);
                                off += n;
                            }
                            if n_learner > 0 {
                                for (j, &a) in learner_idx.iter().enumerate() {
                                    learner_obs[j * d..(j + 1) * d].copy_from_slice(&obs_buf[a * d..(a + 1) * d]);
                                }
                                match pol.forward(&learner_obs, n_learner, d) {
                                    Ok((_, lv)) => out.last_values.copy_from_slice(&lv),
                                    Err(e) => {
                                        let _ = otx.send(WorkerOut::err(e));
                                        continue;
                                    }
                                }
                            }
                            let _ = otx.send(WorkerOut::collect(out));
                        }
                    }
                }
            });
            workers.push(Worker {
                tx: ctx,
                rx: orx,
                handle: Some(handle),
                num_agents: worker_agents,
                num_arenas: count,
            });
        }

        MultiEngine {
            num_agents: sizes.iter().map(|&(b, o)| b + o).sum(),
            num_arenas,
            // V1 has no flat obs: obs_size 0 mirrors schema/v1.toml's
            // obs_size = 0 (the lib.rs getter passes it straight through).
            obs_size: match obs_mode { ObsMode::V0 => OBS_SIZE, ObsMode::V1 => 0 },
            action_count: match (obs_mode, act_table) {
                (ObsMode::V0, _) => crate::actions::TABLE_SIZE,
                (ObsMode::V1, ActionTableKind::V1) => crate::actions::TABLE_SIZE_V1,
                (ObsMode::V1, ActionTableKind::V1Air) => crate::actions::TABLE_SIZE_V1_AIR,
            },
            obs_mode,
            emit_v0_obs,
            workers,
            debug_policy: None,
            sizes,
            opponent_slots: 0,
            foreign_slots: 0,
        }
    }

    pub fn reset_into(&mut self, obs: &mut [f32]) {
        for w in &self.workers {
            w.tx.send(Cmd::Reset).unwrap();
        }
        let mut off = 0;
        for w in &self.workers {
            let out = w.rx.recv().unwrap();
            obs[off * OBS_SIZE..(off + w.num_agents) * OBS_SIZE].copy_from_slice(&out.obs);
            off += w.num_agents;
        }
    }

    pub fn step_into(
        &mut self,
        actions: &[i64],
        obs: &mut [f32],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        final_obs: &mut [f32],
    ) -> Result<(), String> {
        if actions.len() != self.num_agents {
            return Err(format!("expected {} actions, got {}", self.num_agents, actions.len()));
        }
        if let Some(bad) = actions.iter().find(|&&a| a < 0 || a as usize >= self.action_count) {
            return Err(format!("action index {bad} out of range [0, {})", self.action_count));
        }
        let mut off = 0;
        for w in &self.workers {
            w.tx.send(Cmd::Step(actions[off..off + w.num_agents].to_vec())).unwrap();
            off += w.num_agents;
        }
        off = 0;
        for w in &self.workers {
            let out = w.rx.recv().unwrap();
            let (s, e) = (off, off + w.num_agents);
            obs[s * OBS_SIZE..e * OBS_SIZE].copy_from_slice(&out.obs);
            rewards[s..e].copy_from_slice(&out.rewards);
            terminated[s..e].copy_from_slice(&out.terminated);
            truncated[s..e].copy_from_slice(&out.truncated);
            final_obs[s * OBS_SIZE..e * OBS_SIZE].copy_from_slice(&out.final_obs);
            off += w.num_agents;
        }
        Ok(())
    }

    /// Broadcasts `Cmd::SetWeights` to every worker (each builds its own `MlpPolicy`
    /// for Task 4's on-worker rollout) and, on success, also builds a policy copy held
    /// directly on `MultiEngine` for `debug_policy_forward`.
    ///
    /// Design choice: `debug_policy_forward` needs a synchronous request/response call
    /// from Python for parity testing. The brief offered either a worker-0 round-trip
    /// (new `Cmd::DebugForward` + channel plumbing) or a `MultiEngine`-held policy copy
    /// evaluated on the calling thread. We chose the held copy: it needs no new Cmd
    /// variant or channel wiring, keeps the debug path off the worker channels
    /// entirely (so it can never race with in-flight Step/Reset), and candle's CPU
    /// forward pass is cheap enough that duplicating the weights once per set_weights
    /// call is a non-issue. The tradeoff is it technically evaluates "the trainer's
    /// weights" rather than literally "worker 0's net", but since every worker gets
    /// the identical broadcast weights, the two are equivalent in practice.
    /// Dispatches on the `NetWeights` variant, which must match the engine's
    /// `obs_mode` (v0 schema -> `NetWeights::V0`, v1 -> `NetWeights::V1`).
    /// Both variants are validated on the calling thread BEFORE broadcasting
    /// (v0 by building the debug `MlpPolicy` copy it also keeps, v1 by a
    /// throwaway `EntityPolicy` build — no debug copy is kept because
    /// `debug_policy_forward`'s flat-obs contract is v0-only).
    pub fn set_weights(&mut self, weights: NetWeights) -> Result<(), String> {
        let arc = Arc::new(weights);
        let debug_policy = match (self.obs_mode, &*arc) {
            (ObsMode::V0, NetWeights::V0(pw)) => Some(MlpPolicy::new(pw)?),
            (ObsMode::V1, NetWeights::V1 { raw, heads }) => {
                let pol = EntityPolicy::new(raw, *heads)?;
                // CROSS-TABLE GUARD. schema.version is 1 for BOTH v1 action
                // tables, so a 104-row v1-air checkpoint loaded against
                // schema/v1.toml (92 rows) passes every version check and then
                // samples action index 92..104 into a 92-row decode table --
                // an out-of-bounds panic deep inside a worker thread, minutes
                // in, with no hint of the real cause. One Engine binds ONE
                // table; cross-table evaluation needs one engine per side.
                // Fail here, on the calling thread, naming both widths.
                if pol.table_size() != self.action_count {
                    return Err(format!(
                        "action-table mismatch: this engine's schema selects a \
                         {}-row table but the state dict carries a {}-row \
                         action_table. schema_version is 1 for BOTH v1 tables, so \
                         the schema PATH is what distinguishes them -- use \
                         schema/v1.toml for a 92-row checkpoint and \
                         schema/v1_air.toml for a 104-row one. A v1.1 checkpoint \
                         and a v1-air checkpoint cannot share one engine.",
                        self.action_count,
                        pol.table_size()
                    ));
                }
                None
            }
            (ObsMode::V0, NetWeights::V1 { .. }) => {
                return Err("v1 state dict given to a v0-schema engine".into());
            }
            (ObsMode::V1, NetWeights::V0(_)) => {
                return Err("v0 state dict given to a v1-schema engine".into());
            }
            (_, NetWeights::Foreign { .. }) => {
                return Err("foreign state dict given to set_weights; a foreign bot \
                            can only be an OPPONENT (set_foreign_opponents)"
                    .into());
            }
        };
        for w in &self.workers {
            w.tx.send(Cmd::SetWeights(arc.clone())).map_err(|e| e.to_string())?;
        }
        let mut first_err: Option<String> = None;
        for w in &self.workers {
            let out = w.rx.recv().map_err(|e| e.to_string())?;
            if let Some(e) = out.error {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        if let Some(p) = debug_policy {
            self.debug_policy = Some(p);
        }
        Ok(())
    }

    /// Broadcasts `Cmd::SetOpponents` to every worker (each rebuilds its own
    /// `opponents: Vec<MlpPolicy>` from `weights`, indexed by slot) using the same
    /// drain-all discipline as `set_weights` (send to all, then recv from all — a
    /// worker error must not desync a sibling's reply channel). `weights.len()` (0..=8;
    /// the `> 8` check lives in lib.rs, matching the brief's division of labor) becomes
    /// the new `opponent_slots` bound that `collect`'s assignment validation checks
    /// against. An empty `weights` (Python's `set_opponents([])`) clears every worker's
    /// slots and resets `opponent_slots` to 0.
    pub fn set_opponents(&mut self, weights: Vec<NetWeights>) -> Result<(), String> {
        // Every slot's variant must match the engine's obs mode (workers
        // then never see a mixed/mismatched slot list).
        for (slot, w) in weights.iter().enumerate() {
            match (self.obs_mode, w) {
                (ObsMode::V0, NetWeights::V0(_)) => {}
                (ObsMode::V1, NetWeights::V1 { raw, .. }) => {
                    // Same cross-table guard as `set_weights`, and it has to be
                    // here too because MatchRunner.play drives BOTH sides
                    // through ONE engine (`set_weights(a)` + `set_opponents([b])`).
                    // A 104-row v1-air opponent in a 92-row engine would sample
                    // an index past the end of the decode table. Read the width
                    // off the raw buffer rather than building an EntityPolicy:
                    // this path never needed a full construction and a shape
                    // check is what is actually at stake.
                    if let Some((_, shape)) = raw.get("action_table") {
                        if shape.len() == 2 && shape[0] != self.action_count {
                            return Err(format!(
                                "opponent slot {slot}: action-table mismatch -- this \
                                 engine's schema selects a {}-row table but the \
                                 opponent state dict carries a {}-row action_table. \
                                 schema_version is 1 for BOTH v1 tables, so a v1.1 \
                                 checkpoint and a v1-air checkpoint cannot be played \
                                 against each other in one engine; run each side on \
                                 its own schema.",
                                self.action_count, shape[0]
                            ));
                        }
                    }
                }
                (ObsMode::V0, NetWeights::V1 { .. }) => {
                    return Err("v1 opponent state dict given to a v0-schema engine".into());
                }
                (ObsMode::V1, NetWeights::V0(_)) => {
                    return Err("v0 opponent state dict given to a v1-schema engine".into());
                }
                (_, NetWeights::Foreign { .. }) => {
                    return Err("foreign state dict given to set_opponents; use \
                                set_foreign_opponents (separate slot space)"
                        .into());
                }
            }
        }
        let n = weights.len();
        let arc = Arc::new(weights);
        for w in &self.workers {
            w.tx.send(Cmd::SetOpponents(arc.clone())).map_err(|e| e.to_string())?;
        }
        let mut first_err: Option<String> = None;
        for w in &self.workers {
            let out = w.rx.recv().map_err(|e| e.to_string())?;
            if let Some(e) = out.error {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        self.opponent_slots = n;
        Ok(())
    }

    /// Load FOREIGN (ported community bot) opponents into their own slot space.
    /// Addressed in a collect assignment as `-(slot) - 2`. Unlike `set_opponents`
    /// these are obs-mode independent: a foreign bot builds its own observation
    /// from the raw GameState, so it works regardless of our schema version.
    pub fn set_foreign_opponents(&mut self, weights: Vec<NetWeights>) -> Result<(), String> {
        for w in &weights {
            if !matches!(w, NetWeights::Foreign { .. }) {
                return Err("set_foreign_opponents requires foreign state dicts".into());
            }
        }
        let n = weights.len();
        let arc = Arc::new(weights);
        for w in &self.workers {
            w.tx.send(Cmd::SetForeignOpponents(arc.clone())).map_err(|e| e.to_string())?;
        }
        let mut first_err: Option<String> = None;
        for w in &self.workers {
            let out = w.rx.recv().map_err(|e| e.to_string())?;
            if let Some(e) = out.error {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        self.foreign_slots = n;
        Ok(())
    }

    /// Fans `Cmd::Collect { steps, assignment }` to every worker (drain-all: send to
    /// all, then recv from all — same pattern as `set_weights`) and interleaves the
    /// per-worker `CollectOut`s (already learner-row-only, see the worker's
    /// `Cmd::Collect` arm) into global `(T, N_learner, ...)` buffers. Worker order
    /// (and, within a worker, learner-row order) matches `reset_into`/`step_into`'s
    /// full agent order minus the filtered-out opponent-orange rows.
    ///
    /// `assignment` is validated here, once, before any Cmd is sent: length must
    /// match `num_arenas`, every value must be `-1` or a currently-set opponent slot
    /// `[0, opponent_slots)`, and the resulting learner-row count must be nonzero
    /// (an all-opponent config where every arena's blue side is also opponent-driven
    /// would otherwise silently produce empty buffers downstream). `None` from Python
    /// is materialized by lib.rs into `vec![-1; num_arenas]` before this ever sees
    /// it — the legacy path and this path are the same code, not a branch.
    pub fn collect(&mut self, steps: usize, assignment: Arc<Vec<i32>>) -> Result<CollectOut, String> {
        if assignment.len() != self.num_arenas {
            return Err(format!(
                "arena_opponents length {} != num_arenas {}", assignment.len(), self.num_arenas
            ));
        }
        for &k in assignment.iter() {
            // k == -1 self-play; k >= 0 native opponent slot; k <= -2 FOREIGN slot
            // (-k - 2), a separate slot space filled by set_foreign_opponents.
            let bad = if k == -1 {
                false
            } else if k <= -2 {
                ((-k - 2) as usize) >= self.foreign_slots
            } else {
                k as usize >= self.opponent_slots
            };
            if bad {
                return Err(format!(
                    "arena_opponents value {k} out of range: expected -1, a native slot in \
                     [0, {}), or a foreign slot encoded as -(f)-2 for f in [0, {})",
                    self.opponent_slots, self.foreign_slots
                ));
            }
        }
        // Only self-play arenas (-1) contribute their orange cars as learner rows;
        // both native-opponent and foreign arenas contribute blue only.
        let learner_count: usize = self.sizes.iter().zip(assignment.iter())
            .map(|(&(b, o), &k)| if k == -1 { b + o } else { b })
            .sum();
        if learner_count == 0 {
            return Err("arena_opponents assignment yields no learner agents".into());
        }

        for w in &self.workers {
            w.tx.send(Cmd::Collect { steps, assignment: assignment.clone() }).map_err(|e| e.to_string())?;
        }
        let mut worker_outs = Vec::with_capacity(self.workers.len());
        for w in &self.workers {
            worker_outs.push(w.rx.recv().map_err(|e| e.to_string())?);
        }
        if let Some(e) = worker_outs.iter_mut().find_map(|o| o.error.take()) {
            return Err(e);
        }

        match self.obs_mode {
            ObsMode::V0 => {
                let d = self.obs_size;
                let mut merged = CollectOut::zeros(steps, learner_count, d);
                let mut off = 0usize;
                for out in worker_outs.into_iter() {
                    let co = out.collect.expect("collect payload missing on worker success");
                    let n = co.learner_agents;
                    for t in 0..steps {
                        let o_src = t * n * d..(t + 1) * n * d;
                        let o_dst = t * learner_count * d + off * d;
                        merged.obs[o_dst..o_dst + n * d].copy_from_slice(&co.obs[o_src]);

                        let s_src = t * n..(t + 1) * n;
                        let s_dst = t * learner_count + off;
                        merged.actions[s_dst..s_dst + n].copy_from_slice(&co.actions[s_src.clone()]);
                        merged.logprobs[s_dst..s_dst + n].copy_from_slice(&co.logprobs[s_src.clone()]);
                        merged.values[s_dst..s_dst + n].copy_from_slice(&co.values[s_src.clone()]);
                        merged.rewards[s_dst..s_dst + n].copy_from_slice(&co.rewards[s_src.clone()]);
                        merged.terminated[s_dst..s_dst + n].copy_from_slice(&co.terminated[s_src.clone()]);
                        merged.truncated[s_dst..s_dst + n].copy_from_slice(&co.truncated[s_src.clone()]);
                        merged.final_values[s_dst..s_dst + n].copy_from_slice(&co.final_values[s_src]);
                    }
                    merged.last_values[off..off + n].copy_from_slice(&co.last_values);
                    off += n;
                }
                Ok(merged)
            }
            ObsMode::V1 => {
                // Same interleave as v0, with the flat obs replaced by the
                // four entity buffers (+ optional obs_v0) at their own
                // per-agent widths.
                fn ilv<T: Copy>(dst: &mut [T], src: &[T], t: usize, n_total: usize, off: usize, n: usize, k: usize) {
                    let d_start = (t * n_total + off) * k;
                    dst[d_start..d_start + n * k].copy_from_slice(&src[t * n * k..(t + 1) * n * k]);
                }
                let mut merged = CollectOut::zeros_v1(steps, learner_count, self.emit_v0_obs);
                let mut off = 0usize;
                let mut fwd_rows = 0usize;
                for out in worker_outs.into_iter() {
                    let co = out.collect.expect("collect payload missing on worker success");
                    let n = co.learner_agents;
                    for t in 0..steps {
                        ilv(&mut merged.ents, &co.ents, t, learner_count, off, n, MAX_ENT * ENT_FEAT);
                        ilv(&mut merged.mask, &co.mask, t, learner_count, off, n, MAX_ENT);
                        ilv(&mut merged.query, &co.query, t, learner_count, off, n, Q_FEAT);
                        ilv(&mut merged.prev, &co.prev, t, learner_count, off, n, PREV_ACTIONS);
                        if self.emit_v0_obs {
                            ilv(&mut merged.obs_v0, &co.obs_v0, t, learner_count, off, n, OBS_SIZE);
                        }
                        ilv(&mut merged.actions, &co.actions, t, learner_count, off, n, 1);
                        ilv(&mut merged.logprobs, &co.logprobs, t, learner_count, off, n, 1);
                        ilv(&mut merged.values, &co.values, t, learner_count, off, n, 1);
                        ilv(&mut merged.rewards, &co.rewards, t, learner_count, off, n, 1);
                        ilv(&mut merged.terminated, &co.terminated, t, learner_count, off, n, 1);
                        ilv(&mut merged.truncated, &co.truncated, t, learner_count, off, n, 1);
                        ilv(&mut merged.final_values, &co.final_values, t, learner_count, off, n, 1);
                    }
                    merged.last_values[off..off + n].copy_from_slice(&co.last_values);
                    // Same interleave as last_values: workers own contiguous
                    // arena ranges, so concatenating in worker order reproduces
                    // the global learner-row order exactly.
                    merged.learner_team_size[off..off + n]
                        .copy_from_slice(&co.learner_team_size);
                    fwd_rows += co.forward_rows;
                    off += n;
                }
                merged.forward_rows = fwd_rows;
                Ok(merged)
            }
        }
    }

    /// Per-arena team size (1/2/3) in global arena order. See `Engine.team_sizes`
    /// in lib.rs for why the trainer needs it.
    pub fn team_sizes(&self) -> Vec<usize> {
        self.sizes.iter().map(|&(b, o)| b.max(o)).collect()
    }

    /// Sum every worker's arenas' per-term reward counters, resetting them (E9).
    ///
    /// Deliberately its own Cmd rather than a field on `CollectOut`: the collect
    /// buffers' shapes are load-bearing for the merge arithmetic and for the
    /// determinism regression tests, and telemetry must never be able to perturb
    /// them.
    pub fn reward_terms(&mut self) -> Result<[f64; crate::reward::N_TERMS], String> {
        for w in &self.workers {
            w.tx.send(Cmd::RewardTerms).map_err(|e| e.to_string())?;
        }
        let mut sums = [0.0f64; crate::reward::N_TERMS];
        let mut first_err: Option<String> = None;
        for w in &self.workers {
            let out = w.rx.recv().map_err(|e| e.to_string())?;
            if let Some(e) = out.error {
                if first_err.is_none() {
                    first_err = Some(e);
                }
                continue;
            }
            if let Some(t) = out.terms {
                for (s, v) in sums.iter_mut().zip(t) {
                    *s += v;
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(sums),
        }
    }

    /// Runs `obs` (row-major batch*obs_size) through the `MultiEngine`-held policy
    /// copy built by the most recent `set_weights` call. Evaluated on the calling
    /// (Python) thread — no worker round-trip. See `set_weights` for why.
    pub fn debug_policy_forward(
        &self,
        obs: &[f32],
        batch: usize,
        obs_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let policy = self.debug_policy.as_ref().ok_or("set_weights has not been called yet")?;
        policy.forward(obs, batch, obs_dim)
    }

    /// Maps the global `arena_idx` to (worker, local_idx) using the same contiguous
    /// even-split assignment as construction, sends `Cmd::Debug`, and returns the
    /// worker's JSON state dump + that arena's obs (all agents) + agent count.
    pub fn debug_arena(&mut self, arena_idx: usize) -> Result<(String, Vec<f32>, usize), String> {
        let mut base = 0usize;
        for w in &self.workers {
            if arena_idx < base + w.num_arenas {
                let local_idx = arena_idx - base;
                w.tx.send(Cmd::Debug { local_idx }).map_err(|e| e.to_string())?;
                let out = w.rx.recv().map_err(|e| e.to_string())?;
                let agents = out.obs.len() / OBS_SIZE;
                let json = out.debug_json.ok_or_else(|| "worker returned no debug json".to_string())?;
                return Ok((json, out.obs, agents));
            }
            base += w.num_arenas;
        }
        Err(format!("arena_idx {arena_idx} out of range [0, {base})"))
    }
}

impl Drop for MultiEngine {
    fn drop(&mut self) {
        for w in &self.workers {
            let _ = w.tx.send(Cmd::Shutdown);
        }
        for w in &mut self.workers {
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_largest_remainder_deterministic() {
        assert_eq!(allocate_team_sizes(4, [1.0, 0.0, 0.0]), vec![1, 1, 1, 1]);
        // 192 arenas at [0.5, 0.3, 0.2] -> 96/58/38 (0.5*192=96, 57.6->58 via remainder, 38.4->38)
        let s = allocate_team_sizes(192, [0.5, 0.3, 0.2]);
        assert_eq!(s.iter().filter(|&&x| x == 1).count(), 96);
        assert_eq!(s.iter().filter(|&&x| x == 2).count(), 58);
        assert_eq!(s.iter().filter(|&&x| x == 3).count(), 38);
        assert_eq!(s.len(), 192);
        // blocks are ordered 1s, 2s, 3s
        let mut sorted = s.clone();
        sorted.sort_unstable();
        assert_eq!(s, sorted);
        // exact division stays exact
        assert_eq!(allocate_team_sizes(4, [0.5, 0.25, 0.25]), vec![1, 1, 2, 3]);
    }
}
