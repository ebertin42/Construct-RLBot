use crate::{
    actions,
    curriculum::CurriculumConfig,
    episode::{EpisodeArena, StepFlags},
    obs::OBS_SIZE,
    policy::{LayerWeights, MlpPolicy, PolicyWeights},
    reward::RewardConfig,
    sampler::{sample_categorical, Pcg32},
    schema::Schema,
};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

enum Cmd {
    Reset,
    Step(Vec<i64>),
    Debug { local_idx: usize },
    SetWeights(Arc<PolicyWeights>),
    SetOpponents(Arc<Vec<PolicyWeights>>),
    // `assignment` is the FULL (global, length num_arenas) opponent assignment;
    // each worker slices its own `[global_base..global_base+count)` range out of
    // it (see the Cmd::Collect arm). Legacy calls (Python `arena_opponents=None`)
    // are materialized by lib.rs into `vec![-1; num_arenas]` before this is ever
    // constructed — there is no separate "no assignment" variant, so the legacy
    // path and the opponent path run through literally the same code (byte-
    // identity regression test pins this).
    Collect { steps: usize, assignment: Arc<Vec<i32>> },
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
}

impl CollectOut {
    fn zeros(steps: usize, agents: usize, obs_dim: usize) -> Self {
        CollectOut {
            obs: vec![0.0; steps * agents * obs_dim],
            actions: vec![0; steps * agents],
            logprobs: vec![0.0; steps * agents],
            values: vec![0.0; steps * agents],
            rewards: vec![0.0; steps * agents],
            terminated: vec![false; steps * agents],
            truncated: vec![false; steps * agents],
            final_values: vec![0.0; steps * agents],
            last_values: vec![0.0; agents],
            learner_agents: agents,
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
}

impl MultiEngine {
    /// `sizes[arena] = (blue, orange)` cars-per-team for that arena, one entry per
    /// arena (so `sizes.len()` is the arena count). Uniform legacy construction is
    /// `vec![(blue, orange); num_arenas]`; mixed-team-size construction (Task 2) maps
    /// `engine::allocate_team_sizes` output `s` to `(s, s)` per arena. Kept as pairs
    /// (rather than a single size) so the asymmetric case (e.g. blue != orange, used
    /// by tests) keeps working uniformly with the mixed-size path.
    pub fn new(
        sizes: Vec<(usize, usize)>,
        schema: Schema,
        reward_cfg: RewardConfig,
        seed: u32,
        num_threads: usize,
        curriculum: Option<CurriculumConfig>,
    ) -> Self {
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
                        EpisodeArena::new_with_curriculum(b, o, sch.tick_skip, cfg.clone(),
                                          sch.normalization.clone(), seed.wrapping_add((global_base + i) as u32),
                                          curr.clone())
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
                // Opponent-policy slots (indexed by the `k >= 0` values in a
                // Collect assignment), rebuilt wholesale on every SetOpponents.
                // Empty until set_opponents is called — legacy (all-self-play)
                // collects never index into this.
                let mut opponents: Vec<MlpPolicy> = Vec::new();
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
                            };
                            ar.write_obs(&mut out.obs);
                            out.debug_json = Some(ar.debug_state_json());
                            let _ = otx.send(out);
                        }
                        Cmd::SetWeights(w) => {
                            match MlpPolicy::new(&w) {
                                Ok(p) => {
                                    policy = Some(p);
                                    let _ = otx.send(WorkerOut::ack());
                                }
                                Err(e) => {
                                    let _ = otx.send(WorkerOut::err(e));
                                }
                            }
                        }
                        Cmd::SetOpponents(ws) => {
                            let mut built = Vec::with_capacity(ws.len());
                            let mut build_err: Option<String> = None;
                            for w in ws.iter() {
                                match MlpPolicy::new(w) {
                                    Ok(p) => built.push(p),
                                    Err(e) => { build_err = Some(e); break; }
                                }
                            }
                            match build_err {
                                Some(e) => { let _ = otx.send(WorkerOut::err(e)); }
                                None => {
                                    opponents = built;
                                    let _ = otx.send(WorkerOut::ack());
                                }
                            }
                        }
                        Cmd::Collect { steps, assignment } => {
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
            obs_size: OBS_SIZE,
            action_count: crate::actions::TABLE_SIZE,
            workers,
            debug_policy: None,
            sizes,
            opponent_slots: 0,
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
    pub fn set_weights(&mut self, weights: PolicyWeights) -> Result<(), String> {
        let arc = Arc::new(weights);
        let debug_policy = MlpPolicy::new(&arc)?;
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
        self.debug_policy = Some(debug_policy);
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
    pub fn set_opponents(&mut self, weights: Vec<PolicyWeights>) -> Result<(), String> {
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
            if k < -1 || (k >= 0 && k as usize >= self.opponent_slots) {
                return Err(format!(
                    "arena_opponents value {k} out of range: expected -1 or an opponent slot in [0, {})",
                    self.opponent_slots
                ));
            }
        }
        let learner_count: usize = self.sizes.iter().zip(assignment.iter())
            .map(|(&(b, o), &k)| if k < 0 { b + o } else { b })
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
