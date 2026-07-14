use crate::{
    episode::{EpisodeArena, StepFlags},
    obs::OBS_SIZE,
    policy::{LayerWeights, MlpPolicy, PolicyWeights},
    reward::RewardConfig,
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
    Shutdown,
}

/// Worker -> main-thread reply. `Step`/`Reset`/`Debug` populate the step-shaped
/// buffers and leave `error` as `None`; `SetWeights` leaves the buffers empty and
/// uses `error` to signal ack (`None`) vs failure (`Some(msg)`). One struct (rather
/// than a response enum per Cmd) keeps the worker's single `Sender<WorkerOut>`
/// channel type unchanged and gives Task 4's `Cmd::Collect` the same ack/err slot
/// to reuse alongside whatever payload fields it adds.
struct WorkerOut {
    obs: Vec<f32>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    final_obs: Vec<f32>,
    debug_json: Option<String>,
    error: Option<String>,
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
        }
    }

    fn ack() -> Self {
        WorkerOut::empty()
    }

    fn err(msg: String) -> Self {
        WorkerOut { error: Some(msg), ..WorkerOut::empty() }
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
    Ok(PolicyWeights { trunk, policy, value })
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
    pub fn new(
        num_arenas: usize,
        blue: usize,
        orange: usize,
        schema: Schema,
        reward_cfg: RewardConfig,
        seed: u32,
        num_threads: usize,
    ) -> Self {
        let threads = if num_threads == 0 {
            std::thread::available_parallelism().map(|n| n.get().saturating_sub(2).max(1)).unwrap_or(4)
        } else {
            num_threads
        }
        .min(num_arenas);
        let per_agent = blue + orange;

        // distribute arenas round-robin-contiguously over threads
        let mut workers = Vec::with_capacity(threads);
        let mut assigned = 0usize;
        for t in 0..threads {
            let count = (num_arenas - assigned) / (threads - t); // even split
            assigned += count;
            let (ctx, crx) = channel::<Cmd>();
            let (otx, orx) = channel::<WorkerOut>();
            let (sch, cfg) = (schema.clone(), reward_cfg.clone());
            // seed by GLOBAL arena index so rollouts are invariant to num_threads
            let global_base = assigned - count;
            let handle = std::thread::spawn(move || {
                // arenas created inside the worker thread
                let mut arenas: Vec<EpisodeArena> = (0..count)
                    .map(|i| {
                        EpisodeArena::new(blue, orange, sch.tick_skip, cfg.clone(),
                                          sch.normalization.clone(), seed.wrapping_add((global_base + i) as u32))
                    })
                    .collect();
                let agents = count * per_agent;
                // Holds the worker's own MlpPolicy, set via Cmd::SetWeights. Unused by
                // Reset/Step/Debug today; Task 4's Cmd::Collect (on-worker rollout with
                // policy-driven actions) will read it. `debug_policy_forward` (Task 3)
                // does not route through here — see MultiEngine::debug_policy_forward.
                #[allow(unused_assignments, unused_variables)]
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
                            };
                            let mut a_off = 0;
                            let mut flags = vec![StepFlags::default(); per_agent];
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
                            };
                            ar.write_obs(&mut out.obs);
                            out.debug_json = Some(ar.debug_state_json());
                            let _ = otx.send(out);
                        }
                        Cmd::SetWeights(w) => {
                            match MlpPolicy::new(&w) {
                                Ok(p) => {
                                    #[allow(unused_assignments)]
                                    {
                                        policy = Some(p);
                                    }
                                    let _ = otx.send(WorkerOut::ack());
                                }
                                Err(e) => {
                                    let _ = otx.send(WorkerOut::err(e));
                                }
                            }
                        }
                    }
                }
            });
            workers.push(Worker {
                tx: ctx,
                rx: orx,
                handle: Some(handle),
                num_agents: count * per_agent,
                num_arenas: count,
            });
        }

        MultiEngine {
            num_agents: num_arenas * per_agent,
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
