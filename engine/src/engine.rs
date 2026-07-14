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
    Collect { steps: usize },
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
    pub obs_size: usize,
    pub action_count: usize,
    // Debug-forward policy copy — see `set_weights`/`debug_policy_forward` doc comments
    // for why this lives here instead of routing through a worker.
    debug_policy: Option<MlpPolicy>,
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
                        Cmd::Collect { steps } => {
                            let Some(pol) = policy.as_ref() else {
                                let _ = otx.send(WorkerOut::err("collect before set_weights".into()));
                                continue;
                            };
                            let d = OBS_SIZE;
                            let action_count = actions::TABLE_SIZE;
                            let mut out = CollectOut::zeros(steps, agents, d);
                            let mut obs_buf = vec![0f32; agents * d];
                            let mut rew_buf = vec![0f32; max_arena_agents];
                            let mut flag_buf = vec![StepFlags::default(); max_arena_agents];
                            let mut fin_buf = vec![0f32; agents * d];

                            let mut worker_err: Option<String> = None;
                            'rounds: for t in 0..steps {
                                // 1. obs for all my arenas
                                let mut off = 0;
                                for ar in arenas.iter_mut() {
                                    let n = ar.num_agents() * d;
                                    ar.write_obs(&mut obs_buf[off..off + n]);
                                    off += n;
                                }
                                out.obs[t * agents * d..(t + 1) * agents * d].copy_from_slice(&obs_buf);

                                // 2. ONE batched forward for the whole worker (all agents of
                                // all this worker's arenas). NOTE: this makes float outputs
                                // (values/logits) depend on the worker's batch size, which
                                // depends on how arenas are split across threads — candle's
                                // CPU gemm rounds the *same* input row differently at
                                // different batch sizes (~1e-7 in the logits, verified
                                // empirically). Cross-thread-count determinism is therefore
                                // NOT provided: those ~1e-7 logit differences occasionally
                                // flip which CDF bucket a sample lands in, actions diverge,
                                // and trajectories separate entirely — so no useful
                                // cross-thread-count guarantee exists at any tolerance.
                                // The determinism contract is exact reproducibility for a
                                // fixed (seed, num_arenas, num_threads) config only. A
                                // per-arena-forward variant that WAS thread-count exact was
                                // tried and reverted: batch=per_agent gemm calls are
                                // overhead-dominated (18k env-steps/s vs the Python path's
                                // ~45k at 96 arenas).
                                let (logits, values) = match pol.forward(&obs_buf, agents, d) {
                                    Ok(x) => x,
                                    Err(e) => { worker_err = Some(e); break 'rounds; }
                                };
                                out.values[t * agents..(t + 1) * agents].copy_from_slice(&values);

                                // 3. sample per agent — agent a belongs to arena a_to_arena[a]
                                // (arenas may hold different agent counts now, in worker
                                // order); each arena's agents draw from that arena's own rng
                                // in blue-then-orange order, never shared across arenas.
                                let mut acts = vec![0i64; agents];
                                for a in 0..agents {
                                    let row = &logits[a * action_count..(a + 1) * action_count];
                                    let (idx, lp) = sample_categorical(row, &mut rngs[a_to_arena[a]]);
                                    acts[a] = idx as i64;
                                    out.actions[t * agents + a] = idx as i64;
                                    out.logprobs[t * agents + a] = lp;
                                }

                                // 4. step arenas, collect rewards/flags/final_obs
                                let mut aoff = 0;
                                for ar in arenas.iter_mut() {
                                    let n = ar.num_agents();
                                    ar.step(
                                        &acts[aoff..aoff + n],
                                        &mut rew_buf[..n],
                                        &mut flag_buf[..n],
                                        &mut fin_buf[aoff * d..(aoff + n) * d],
                                    );
                                    for i in 0..n {
                                        out.rewards[t * agents + aoff + i] = rew_buf[i];
                                        out.terminated[t * agents + aoff + i] = flag_buf[i].terminated;
                                        out.truncated[t * agents + aoff + i] = flag_buf[i].truncated;
                                    }
                                    aoff += n;
                                }

                                // 5. final values for done rows (single batched forward over
                                // just the done subset, worker-wide)
                                let done: Vec<usize> = (0..agents)
                                    .filter(|&a| out.terminated[t * agents + a] || out.truncated[t * agents + a])
                                    .collect();
                                if !done.is_empty() {
                                    let mut fobs = vec![0f32; done.len() * d];
                                    for (j, &a) in done.iter().enumerate() {
                                        fobs[j * d..(j + 1) * d].copy_from_slice(&fin_buf[a * d..(a + 1) * d]);
                                    }
                                    match pol.forward(&fobs, done.len(), d) {
                                        Ok((_, fv)) => {
                                            for (j, &a) in done.iter().enumerate() {
                                                out.final_values[t * agents + a] = fv[j];
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

                            // 6. bootstrap values of the post-rollout obs (one batched forward)
                            let mut off = 0;
                            for ar in arenas.iter_mut() {
                                let n = ar.num_agents() * d;
                                ar.write_obs(&mut obs_buf[off..off + n]);
                                off += n;
                            }
                            match pol.forward(&obs_buf, agents, d) {
                                Ok((_, lv)) => out.last_values.copy_from_slice(&lv),
                                Err(e) => {
                                    let _ = otx.send(WorkerOut::err(e));
                                    continue;
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
            obs_size: OBS_SIZE,
            action_count: crate::actions::TABLE_SIZE,
            workers,
            debug_policy: None,
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

    /// Fans `Cmd::Collect { steps }` to every worker (drain-all: send to all, then
    /// recv from all — same pattern as `set_weights`) and interleaves the per-worker
    /// `CollectOut`s into global `(T, N, ...)` buffers. Worker `w`'s agents occupy
    /// contiguous columns `[w_base .. w_base + w_agents)`, same column order
    /// `reset_into`/`step_into` use. Errors (e.g. collect before set_weights) from
    /// any worker are surfaced as `Err`.
    pub fn collect(&mut self, steps: usize) -> Result<CollectOut, String> {
        for w in &self.workers {
            w.tx.send(Cmd::Collect { steps }).map_err(|e| e.to_string())?;
        }
        let mut worker_outs = Vec::with_capacity(self.workers.len());
        for w in &self.workers {
            worker_outs.push(w.rx.recv().map_err(|e| e.to_string())?);
        }
        if let Some(e) = worker_outs.iter_mut().find_map(|o| o.error.take()) {
            return Err(e);
        }

        let d = self.obs_size;
        let agents = self.num_agents;
        let mut merged = CollectOut::zeros(steps, agents, d);
        let mut off = 0usize;
        for (w, out) in self.workers.iter().zip(worker_outs.into_iter()) {
            let n = w.num_agents;
            let co = out.collect.expect("collect payload missing on worker success");
            for t in 0..steps {
                let o_src = t * n * d..(t + 1) * n * d;
                let o_dst = t * agents * d + off * d;
                merged.obs[o_dst..o_dst + n * d].copy_from_slice(&co.obs[o_src]);

                let s_src = t * n..(t + 1) * n;
                let s_dst = t * agents + off;
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
