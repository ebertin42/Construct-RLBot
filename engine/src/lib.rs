use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

pub mod actions;
pub mod ballpred;
pub mod curriculum;
pub mod engine;
pub mod episode;
pub mod obs;
pub mod obs_v1;
pub mod policy;
pub mod policy_v1;
pub mod reset_pool;
pub mod reward;
pub mod sampler;
pub mod schema;
pub mod sim_init;
pub mod viser;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn action_table<'py>(py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
    let t = actions::make_lookup_table();
    let flat: Vec<f32> = t.iter().flatten().copied().collect();
    numpy::ndarray::Array2::from_shape_vec((t.len(), 8), flat).unwrap().into_pyarray(py)
}

/// The v1.1 action table (92 rows: v0's 90 + 2 stall rows) as an [N,8] f32
/// array. The Python `EntityPolicyNet` consumes this as its non-trainable
/// `action_table` buffer (prev-action embedding + dot action head).
#[pyfunction]
fn action_table_v1<'py>(py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
    let t = actions::make_lookup_table_v1();
    let flat: Vec<f32> = t.iter().flatten().copied().collect();
    numpy::ndarray::Array2::from_shape_vec((t.len(), 8), flat).unwrap().into_pyarray(py)
}

#[pyfunction]
fn schema_dict<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let s = crate::schema::Schema::load(path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("version", s.version)?;
    d.set_item("obs_size", s.obs_size)?;
    d.set_item("action_table", s.action_table)?;
    d.set_item("action_count", s.action_count)?;
    d.set_item("tick_skip", s.tick_skip)?;
    d.set_item("pos_norm", s.normalization.pos_norm)?;
    d.set_item("vel_norm", s.normalization.vel_norm)?;
    d.set_item("ang_vel_norm", s.normalization.ang_vel_norm)?;
    Ok(d)
}

// `unsendable`: MultiEngine holds `std::sync::mpsc::Receiver<WorkerOut>` internally
// (per worker), which is Send but not Sync, so pyo3's default Send+Sync pyclass bound
// fails to compile. The actor model itself (worker threads own their arenas; the main
// thread only touches Sender/Receiver ends) is exactly per the brief and is unaffected —
// `unsendable` only pins the *Python-facing* Engine wrapper to the single OS thread that
// constructed it (i.e. the thread holding the GIL when Engine() is called), which matches
// how the trainer contract uses this class (single Python thread drives reset()/step()).
#[pyclass(unsendable)]
struct Engine {
    inner: engine::MultiEngine,
    // Attention head count used when building `EntityPolicy` from a v1
    // state dict (not recoverable from tensor shapes) — `net_heads` kwarg.
    net_heads: usize,
}

/// Schema-vs-compiled-engine validation shared by `Engine::new` and
/// `RenderSession::new`. Version 1 checks the v1 action table (the entity
/// obs meta was already validated by `Schema::load`); every other version
/// keeps the exact legacy v0 check.
fn validate_schema(schema_path: &str, sch: &schema::Schema) -> PyResult<()> {
    if sch.version == 1 {
        if sch.action_count != actions::TABLE_SIZE_V1 || sch.action_table != "construct_92_v1" {
            return Err(PyValueError::new_err(format!(
                "schema {} disagrees with compiled engine (actions {} vs {}, table {:?})",
                schema_path, sch.action_count, actions::TABLE_SIZE_V1, sch.action_table
            )));
        }
        return Ok(());
    }
    if sch.obs_size != obs::OBS_SIZE || sch.action_count != actions::TABLE_SIZE || sch.action_table != "rlgym_lookup_90" {
        return Err(PyValueError::new_err(format!(
            "schema {} disagrees with compiled engine (obs {} vs {}, actions {} vs {}, table {:?})",
            schema_path, sch.obs_size, obs::OBS_SIZE, sch.action_count, actions::TABLE_SIZE, sch.action_table
        )));
    }
    Ok(())
}

impl Engine {
    /// Guard for the Python entry points that only exist on the v0 (flat
    /// obs) path — the v1 contract is collect-only (in-engine inference).
    fn v0_only(&self, what: &str) -> PyResult<()> {
        if self.inner.obs_mode == episode::ObsMode::V1 {
            return Err(PyValueError::new_err(format!(
                "{what} is not supported with a v1 schema (v1 is collect-only; obs are entity tensors)"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (num_arenas=32, blue=1, orange=1, schema_path="schema/v0.toml",
                        reward_config_path="configs/reward_v0.toml", meshes_path=None,
                        seed=0, num_threads=0, team_size_weights=None, curriculum_config_path=None,
                        emit_v0_obs=false, net_heads=4))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_arenas: usize,
        blue: usize,
        orange: usize,
        schema_path: &str,
        reward_config_path: &str,
        meshes_path: Option<&str>,
        seed: u32,
        num_threads: usize,
        team_size_weights: Option<Vec<f64>>,
        curriculum_config_path: Option<&str>,
        emit_v0_obs: bool,
        net_heads: usize,
    ) -> PyResult<Self> {
        sim_init::ensure_init(meshes_path);
        let sch = schema::Schema::load(schema_path).map_err(PyValueError::new_err)?;
        validate_schema(schema_path, &sch)?;
        if emit_v0_obs && sch.version != 1 {
            return Err(PyValueError::new_err(
                "emit_v0_obs requires a v1 schema (v0 collect already emits the v0 obs)",
            ));
        }
        let cfg = reward::RewardConfig::load(reward_config_path).map_err(PyValueError::new_err)?;
        // `sizes[arena] = (blue, orange)`. `None` -> legacy uniform behavior (keeps the
        // asymmetric blue != orange case, e.g. tests' 1v0-style configs, working exactly
        // as before). `Some(w)` -> mixed 1v1/2v2/3v3 arenas via the largest-remainder
        // allocator; blue/orange args are ignored in that case (each mixed arena is size
        // vs size, i.e. (s, s)).
        let sizes: Vec<(usize, usize)> = match team_size_weights {
            Some(w) => {
                if w.len() != 3 || w.iter().any(|x| *x < 0.0) || w.iter().sum::<f64>() <= 0.0 {
                    return Err(PyValueError::new_err(
                        "team_size_weights must be 3 nonnegative floats summing > 0",
                    ));
                }
                engine::allocate_team_sizes(num_arenas, [w[0], w[1], w[2]])
                    .into_iter()
                    .map(|s| (s, s))
                    .collect()
            }
            None => vec![(blue, orange); num_arenas],
        };
        let curriculum = match curriculum_config_path {
            Some(p) => Some(crate::curriculum::CurriculumConfig::load(p).map_err(PyValueError::new_err)?),
            None => None,
        };
        Ok(Engine {
            inner: engine::MultiEngine::new(sizes, sch, cfg, seed, num_threads, curriculum, emit_v0_obs),
            net_heads,
        })
    }

    #[getter]
    fn num_agents(&self) -> usize { self.inner.num_agents }
    #[getter]
    fn obs_size(&self) -> usize { self.inner.obs_size }
    #[getter]
    fn action_count(&self) -> usize { self.inner.action_count }
    #[getter]
    fn obs_mode(&self) -> &'static str {
        match self.inner.obs_mode {
            episode::ObsMode::V0 => "v0",
            episode::ObsMode::V1 => "v1",
        }
    }
    #[getter]
    fn entity_shape(&self) -> (usize, usize) { (obs_v1::MAX_ENT, obs_v1::ENT_FEAT) }
    #[getter]
    fn query_size(&self) -> usize { obs_v1::Q_FEAT }
    #[getter]
    fn prev_len(&self) -> usize { obs_v1::PREV_ACTIONS }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.v0_only("reset")?;
        let (n, d) = (self.inner.num_agents, self.inner.obs_size);
        let mut obs = vec![0.0f32; n * d];
        py.detach(|| self.inner.reset_into(&mut obs));
        Ok(numpy::ndarray::Array2::from_shape_vec((n, d), obs).unwrap().into_pyarray(py))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        self.v0_only("step")?;
        let acts = actions.as_slice()?.to_vec();
        let (n, d) = (self.inner.num_agents, self.inner.obs_size);
        let (mut obs, mut fin) = (vec![0.0f32; n * d], vec![0.0f32; n * d]);
        let mut rew = vec![0.0f32; n];
        let (mut term, mut trunc) = (vec![false; n], vec![false; n]);
        py.detach(|| {
            self.inner.step_into(&acts, &mut obs, &mut rew, &mut term, &mut trunc, &mut fin)
        })
        .map_err(PyValueError::new_err)?;
        Ok((
            numpy::ndarray::Array2::from_shape_vec((n, d), obs).unwrap().into_pyarray(py),
            rew.into_pyarray(py),
            term.into_pyarray(py),
            trunc.into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((n, d), fin).unwrap().into_pyarray(py),
        ))
    }

    /// Loads a PyTorch `state_dict` (as `dict[str, np.float32 ndarray]`) into every
    /// worker's policy plus (v0) the debug-forward copy held on `MultiEngine`.
    /// Dispatches on the schema version the engine was built with: v0 expects
    /// `construct.learn.model.PolicyValueNet` keys (`trunk.{0,2,...}.{weight,bias}`,
    /// `policy_head.*`, `value_head.*`); v1 expects
    /// `construct.learn.model_v1.EntityPolicyNet` keys (see policy_v1.rs's
    /// mapping table) and builds `EntityPolicy` with the constructor's
    /// `net_heads`.
    fn set_weights(&mut self, weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>) -> PyResult<()> {
        let arrays: HashMap<String, (Vec<f32>, Vec<usize>)> = weights.into_iter()
            .map(|(k, v)| {
                let shape = v.shape().to_vec();
                (k, (v.as_array().iter().copied().collect(), shape))
            })
            .collect();
        let nw = match self.inner.obs_mode {
            episode::ObsMode::V0 => {
                engine::NetWeights::V0(engine::parse_state_dict(arrays).map_err(PyValueError::new_err)?)
            }
            episode::ObsMode::V1 => engine::NetWeights::V1 { raw: arrays, heads: self.net_heads },
        };
        self.inner.set_weights(nw).map_err(PyValueError::new_err)
    }

    /// Loads up to 8 opponent-policy `state_dict`s (same conversion/key contract as
    /// `set_weights`) into every worker's `opponents` slots, indexed 0..len by list
    /// position. Rebuilds all slots wholesale each call — `[]` clears them. A
    /// `collect(..., arena_opponents=...)` value `k >= 0` drives that arena's orange
    /// side with `opponents[k]`; see `collect` below.
    fn set_opponents(&mut self, opponents: Vec<HashMap<String, PyReadonlyArrayDyn<'_, f32>>>) -> PyResult<()> {
        if opponents.len() > 8 {
            return Err(PyValueError::new_err("at most 8 opponent slots"));
        }
        let mode = self.inner.obs_mode;
        let heads = self.net_heads;
        let parsed: Vec<engine::NetWeights> = opponents.into_iter()
            .map(|w| {
                let arrays: HashMap<String, (Vec<f32>, Vec<usize>)> = w.into_iter()
                    .map(|(k, v)| {
                        let shape = v.shape().to_vec();
                        (k, (v.as_array().iter().copied().collect(), shape))
                    })
                    .collect();
                match mode {
                    episode::ObsMode::V0 => engine::parse_state_dict(arrays).map(engine::NetWeights::V0),
                    episode::ObsMode::V1 => Ok(engine::NetWeights::V1 { raw: arrays, heads }),
                }
            })
            .collect::<Result<_, _>>()
            .map_err(PyValueError::new_err)?;
        self.inner.set_opponents(parsed).map_err(PyValueError::new_err)
    }

    /// Runs `steps` rounds of on-worker rollout (policy-driven actions, sampled with
    /// each arena's own deterministic Pcg32 — see `engine::MultiEngine::collect`) and
    /// returns a dict of numpy arrays for the trainer: `obs (T,N,94) f32`, `actions
    /// (T,N) i64`, `logprobs`/`values`/`rewards`/`final_values (T,N) f32`,
    /// `terminated`/`truncated (T,N) bool`, `last_values (N,) f32`, plus
    /// `learner_agents` (python int) — the buffer width `N`. `N` is the FULL agent
    /// count when `arena_opponents` is `None` (legacy, every arena self-play — byte-
    /// identical to the pre-league behavior) or shrinks to just the learner-driven
    /// rows (all agents of self-play arenas + blue agents of opponent arenas) when
    /// `arena_opponents[i]` names an opponent slot (`-1` self-play, `k >= 0` slot `k`
    /// set by `set_opponents`) for one or more arenas. Requires `set_weights` to have
    /// been called first (else `ValueError`); a bad `arena_opponents` (wrong length,
    /// unset slot, or an all-opponent assignment with zero learner rows) is also a
    /// `ValueError`, raised before any worker does any work. The gather blocks for
    /// multiple seconds, so it runs under `py.detach` to free the GIL.
    #[pyo3(signature = (steps, arena_opponents=None))]
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        steps: usize,
        arena_opponents: Option<Vec<i32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let assignment = Arc::new(arena_opponents.unwrap_or_else(|| vec![-1; self.inner.num_arenas]));
        let out = py.detach(|| self.inner.collect(steps, assignment)).map_err(PyValueError::new_err)?;
        let (t, n) = (steps, out.learner_agents);

        let dict = PyDict::new(py);
        match self.inner.obs_mode {
            episode::ObsMode::V0 => {
                let d = self.inner.obs_size;
                let obs: Bound<'py, PyArray3<f32>> =
                    numpy::ndarray::Array3::from_shape_vec((t, n, d), out.obs).unwrap().into_pyarray(py);
                dict.set_item("obs", obs)?;
            }
            episode::ObsMode::V1 => {
                let (me, ef) = (obs_v1::MAX_ENT, obs_v1::ENT_FEAT);
                let ents: Bound<'py, PyArray4<f32>> =
                    numpy::ndarray::Array4::from_shape_vec((t, n, me, ef), out.ents).unwrap().into_pyarray(py);
                let mask: Bound<'py, PyArray3<bool>> =
                    numpy::ndarray::Array3::from_shape_vec((t, n, me), out.mask).unwrap().into_pyarray(py);
                let query: Bound<'py, PyArray3<f32>> =
                    numpy::ndarray::Array3::from_shape_vec((t, n, obs_v1::Q_FEAT), out.query).unwrap().into_pyarray(py);
                let prev: Bound<'py, PyArray3<i64>> =
                    numpy::ndarray::Array3::from_shape_vec((t, n, obs_v1::PREV_ACTIONS), out.prev).unwrap().into_pyarray(py);
                dict.set_item("ents", ents)?;
                dict.set_item("mask", mask)?;
                dict.set_item("query", query)?;
                dict.set_item("prev", prev)?;
                if !out.obs_v0.is_empty() {
                    let obs_v0: Bound<'py, PyArray3<f32>> =
                        numpy::ndarray::Array3::from_shape_vec((t, n, obs::OBS_SIZE), out.obs_v0).unwrap().into_pyarray(py);
                    dict.set_item("obs_v0", obs_v0)?;
                }
            }
        }

        let actions: Bound<'py, PyArray2<i64>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.actions).unwrap().into_pyarray(py);
        let logprobs: Bound<'py, PyArray2<f32>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.logprobs).unwrap().into_pyarray(py);
        let values: Bound<'py, PyArray2<f32>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.values).unwrap().into_pyarray(py);
        let rewards: Bound<'py, PyArray2<f32>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.rewards).unwrap().into_pyarray(py);
        let terminated: Bound<'py, PyArray2<bool>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.terminated).unwrap().into_pyarray(py);
        let truncated: Bound<'py, PyArray2<bool>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.truncated).unwrap().into_pyarray(py);
        let final_values: Bound<'py, PyArray2<f32>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.final_values).unwrap().into_pyarray(py);
        let last_values: Bound<'py, PyArray1<f32>> = out.last_values.into_pyarray(py);

        dict.set_item("actions", actions)?;
        dict.set_item("logprobs", logprobs)?;
        dict.set_item("values", values)?;
        dict.set_item("rewards", rewards)?;
        dict.set_item("terminated", terminated)?;
        dict.set_item("truncated", truncated)?;
        dict.set_item("final_values", final_values)?;
        dict.set_item("last_values", last_values)?;
        dict.set_item("learner_agents", n)?;
        Ok(dict)
    }

    /// Runs `obs` (np f32, shape (B, obs_size)) through the policy loaded by the most
    /// recent `set_weights` call. Returns `(logits (B, action_count), values (B,))`.
    /// Used for parity testing against the PyTorch reference net — not part of the
    /// training rollout path (see Task 4's Cmd::Collect for that).
    fn debug_policy_forward<'py>(
        &self,
        py: Python<'py>,
        obs: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>)> {
        self.v0_only("debug_policy_forward")?;
        let shape = obs.shape();
        let (batch, obs_dim) = (shape[0], shape[1]);
        let flat: Vec<f32> = obs.as_array().iter().copied().collect();
        let (logits, values) = self
            .inner
            .debug_policy_forward(&flat, batch, obs_dim)
            .map_err(PyValueError::new_err)?;
        let action_count = logits.len() / batch.max(1);
        Ok((
            numpy::ndarray::Array2::from_shape_vec((batch, action_count), logits)
                .unwrap()
                .into_pyarray(py),
            values.into_pyarray(py),
        ))
    }

    /// JSON dump of arena state + the engine-built obs for all agents of that arena.
    /// Used by tests/python/test_parity.py to check deploy/obs.py against the Rust
    /// obs builder (see Task 12 interfaces for the JSON schema).
    fn debug_state_and_obs<'py>(
        &mut self,
        py: Python<'py>,
        arena_idx: usize,
    ) -> PyResult<(String, Bound<'py, PyArray2<f32>>)> {
        self.v0_only("debug_state_and_obs")?;
        let (json, obs, agents) = self
            .inner
            .debug_arena(arena_idx)
            .map_err(PyValueError::new_err)?;
        Ok((
            json,
            numpy::ndarray::Array2::from_shape_vec((agents, self.inner.obs_size), obs)
                .unwrap()
                .into_pyarray(py),
        ))
    }
}

// `unsendable`: `EpisodeArena` holds a `cxx::UniquePtr<Arena>`, which contains raw
// pointers (`cxx::private::Opaque`) that aren't `Sync`. Same rationale as `Engine`
// above: this only pins the Python-facing wrapper to the constructing thread, which
// matches how `RenderSession` is used (single Python thread drives reset()/step()).
#[pyclass(unsendable)]
struct RenderSession {
    arena: episode::EpisodeArena,
    stream: viser::ViserStream,
    pacer: viser::Pacer,
    num_agents: usize,
    obs_mode: episode::ObsMode,
    // V1 only: the in-session policy for `step_policy` (v1 inference lives
    // in-engine; the Python watch script no longer sees obs tensors).
    policy_v1: Option<policy_v1::EntityPolicy>,
    // V1 only: action-sampling rng for `step_policy` (one stream, agents
    // sampled blue-then-orange like training workers).
    rng: sampler::Pcg32,
    net_heads: usize,
}

impl RenderSession {
    fn v0_only(&self, what: &str) -> PyResult<()> {
        if self.obs_mode == episode::ObsMode::V1 {
            return Err(PyValueError::new_err(format!(
                "{what} is v0-only; with a v1 schema call set_weights(...) then step_policy()"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl RenderSession {
    #[new]
    #[pyo3(signature = (blue=1, orange=1, schema_path="schema/v0.toml",
                        reward_config_path="configs/reward_v0.toml", meshes_path=None, seed=0,
                        net_heads=4))]
    fn new(blue: usize, orange: usize, schema_path: &str, reward_config_path: &str,
           meshes_path: Option<&str>, seed: u32, net_heads: usize) -> PyResult<Self> {
        sim_init::ensure_init(meshes_path);
        let sch = schema::Schema::load(schema_path).map_err(PyValueError::new_err)?;
        validate_schema(schema_path, &sch)?;
        let obs_mode = if sch.version == 1 { episode::ObsMode::V1 } else { episode::ObsMode::V0 };
        let cfg = reward::RewardConfig::load(reward_config_path).map_err(PyValueError::new_err)?;
        let tick_skip = sch.tick_skip;
        let mut arena = episode::EpisodeArena::new_full(
            blue, orange, tick_skip, cfg, sch.normalization, seed, None, obs_mode,
        );
        let mut stream = viser::ViserStream::new().map_err(|e| PyValueError::new_err(e.to_string()))?;
        // real state (valid tick_rate) with cars cleared — see send_flush docs
        let _ = stream.send_flush(arena.game_state());
        Ok(RenderSession {
            arena,
            stream,
            pacer: viser::Pacer::new(tick_skip),
            num_agents: blue + orange,
            obs_mode,
            policy_v1: None,
            rng: sampler::Pcg32::new((seed as u64) * 1_000_003 + 17),
            net_heads,
        })
    }

    #[getter]
    fn obs_size(&self) -> usize {
        match self.obs_mode {
            episode::ObsMode::V0 => obs::OBS_SIZE,
            episode::ObsMode::V1 => 0,
        }
    }
    #[getter]
    fn action_count(&self) -> usize {
        match self.obs_mode {
            episode::ObsMode::V0 => actions::TABLE_SIZE,
            episode::ObsMode::V1 => actions::TABLE_SIZE_V1,
        }
    }
    #[getter]
    fn num_agents(&self) -> usize { self.num_agents }
    #[getter]
    fn obs_mode(&self) -> &'static str {
        match self.obs_mode {
            episode::ObsMode::V0 => "v0",
            episode::ObsMode::V1 => "v1",
        }
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.v0_only("reset")?;
        let (n, d) = (self.num_agents, obs::OBS_SIZE);
        let mut o = vec![0.0f32; n * d];
        self.arena.write_obs(&mut o);
        Ok(numpy::ndarray::Array2::from_shape_vec((n, d), o).unwrap().into_pyarray(py))
    }

    /// V1 only: loads an `EntityPolicyNet` state dict (same key contract as
    /// `Engine.set_weights` in v1 mode) for `step_policy`.
    fn set_weights(&mut self, weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>) -> PyResult<()> {
        if self.obs_mode != episode::ObsMode::V1 {
            return Err(PyValueError::new_err(
                "set_weights is v1-only; the v0 RenderSession is action-driven from Python",
            ));
        }
        let arrays: HashMap<String, (Vec<f32>, Vec<usize>)> = weights.into_iter()
            .map(|(k, v)| {
                let shape = v.shape().to_vec();
                (k, (v.as_array().iter().copied().collect(), shape))
            })
            .collect();
        let p = policy_v1::EntityPolicy::new(&arrays, self.net_heads).map_err(PyValueError::new_err)?;
        self.policy_v1 = Some(p);
        Ok(())
    }

    /// V1 only: one policy-driven env step — builds the entity obs in-engine,
    /// forwards the `set_weights` EntityPolicy, samples actions, steps the
    /// arena, and streams the frame to RLViser. Returns
    /// `(actions (N,) i64, rewards (N,) f32, terminated (N,) bool, truncated (N,) bool)`.
    fn step_policy<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
    )> {
        if self.obs_mode != episode::ObsMode::V1 {
            return Err(PyValueError::new_err("step_policy is v1-only; use step(actions) with v0"));
        }
        let pol = self.policy_v1.as_ref().ok_or_else(|| {
            PyValueError::new_err("step_policy before set_weights")
        })?;
        let n = self.num_agents;
        let (me, ef, qf, pa) =
            (obs_v1::MAX_ENT, obs_v1::ENT_FEAT, obs_v1::Q_FEAT, obs_v1::PREV_ACTIONS);
        let mut ents = vec![0f32; n * me * ef];
        let mut mask = vec![false; n * me];
        let mut query = vec![0f32; n * qf];
        let mut prev = vec![0i64; n * pa];
        self.arena.write_obs_v1(&mut ents, &mut mask, &mut query, &mut prev);
        let (logits, _values) =
            pol.forward(&ents, &mask, &query, &prev, n).map_err(PyValueError::new_err)?;
        let action_count = actions::TABLE_SIZE_V1;
        let mut acts = vec![0i64; n];
        for a in 0..n {
            let row = &logits[a * action_count..(a + 1) * action_count];
            acts[a] = sampler::sample_categorical(row, &mut self.rng).0 as i64;
        }
        let mut rew = vec![0f32; n];
        let mut flags = vec![episode::StepFlags::default(); n];
        let (mut fe, mut fm, mut fq, mut fp) =
            (vec![0f32; n * me * ef], vec![false; n * me], vec![0f32; n * qf], vec![0i64; n * pa]);
        self.arena.step_v1(&acts, &mut rew, &mut flags, &mut fe, &mut fm, &mut fq, &mut fp);
        let gs = self.arena.game_state();
        if episode::state_is_sane(&gs) {
            let _ = self.stream.send_state(gs);
        }
        self.pacer.pace();
        let term: Vec<bool> = flags.iter().map(|f| f.terminated).collect();
        let trunc: Vec<bool> = flags.iter().map(|f| f.truncated).collect();
        Ok((
            acts.into_pyarray(py),
            rew.into_pyarray(py),
            term.into_pyarray(py),
            trunc.into_pyarray(py),
        ))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions_in: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        self.v0_only("step")?;
        let acts = actions_in.as_slice()?.to_vec();
        let (n, d) = (self.num_agents, obs::OBS_SIZE);
        let (mut o, mut fin) = (vec![0.0f32; n * d], vec![0.0f32; n * d]);
        let mut rew = vec![0.0f32; n];
        let mut flags = vec![episode::StepFlags::default(); n];
        self.arena.step(&acts, &mut rew, &mut flags, &mut fin);
        self.arena.write_obs(&mut o);
        let gs = self.arena.game_state();
        // Drop frames from a blowup ramp (huge-but-finite precursors of a NaN
        // state): one such frame permanently poisons rlviser's interpolation.
        if episode::state_is_sane(&gs) {
            let _ = self.stream.send_state(gs);
        }
        self.pacer.pace();
        let term: Vec<bool> = flags.iter().map(|f| f.terminated).collect();
        let trunc: Vec<bool> = flags.iter().map(|f| f.truncated).collect();
        Ok((
            numpy::ndarray::Array2::from_shape_vec((n, d), o).unwrap().into_pyarray(py),
            rew.into_pyarray(py),
            term.into_pyarray(py),
            trunc.into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((n, d), fin).unwrap().into_pyarray(py),
        ))
    }

    fn close(&mut self) {
        self.stream.quit();
    }
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_dict, m)?)?;
    m.add_function(wrap_pyfunction!(action_table, m)?)?;
    m.add_function(wrap_pyfunction!(action_table_v1, m)?)?;
    m.add_class::<Engine>()?;
    m.add_class::<RenderSession>()?;
    Ok(())
}
