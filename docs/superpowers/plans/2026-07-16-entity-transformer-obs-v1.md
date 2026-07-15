# Entity Transformer + Obs v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 94-float MLP policy with a player-count-invariant entity transformer (obs v1, action table v1.1), kickstarted from the current MLP so no training progress is lost.

**Architecture:** Obs v1 is an entity set — ball, ≤6 cars (masked), 6 big pads, 4 ball-prediction entities — each a fixed feature row, plus a self/query row carrying pad timers, prev-5 actions, scoreboard. The net is a pre-LN transformer encoder over entities with self-entity query pooling (Necto/EARL lineage, verified), a ControlsPredictorDot action head (dot(action-embedding, player-embedding) — table-size-independent), a value head, and config-gated aux heads (Lucy-SKG). Rust engine builds the entity tensor + runs candle inference for collect; Python mirrors the exact net for learning. Migration = kickstarting (Schmitt 2018): PPO + annealed on-policy KL to the frozen MLP teacher + value-head regression, so the transformer starts at, not below, the current 1.6 goals/min.

**Tech Stack:** Rust (candle-core/candle-nn 0.11 attention from primitives, rocketsim_rs 0.37 cloned-arena ball prediction), Python (torch pre-LN TransformerEncoder-equivalent, custom — NOT nn.TransformerEncoder, to guarantee weight-name parity with candle), TOML schema v1.

## Global Constraints

- **Schema v1 is a separate file (`schema/v1.toml`)**; v0 checkpoints/paths never change meaning. Engine selects behavior by `schema.version` (0 = legacy MLP path, byte-identical; 1 = entity path). All v0 tests keep passing untouched.
- **Fixed-config determinism contract** (seed, num_arenas, num_threads) applies to the v1 path exactly as v0.
- **Action table v1.1 = the existing 90 rows APPENDED with 2 stall rows** `[0,0,0,1,-1,1,0,1]` and `[0,0,0,-1,1,1,0,1]` (rlgym-tools-verified stall inputs; dodgeDir = (-pitch, yaw+roll) ⇒ yaw=-roll ⇒ zero impulse) → **TABLE_SIZE_V1 = 92**. Append-only: indices 0-89 keep their v0 meaning (kickstart teacher logits map 1:1 onto the first 90 slots). Wavedash needs no entry (hb already auto-set on jump+torque rows); flip reset is a state event, not an input.
- **Net dims are config, not constants** (`[net] d_model, layers, heads, ff`). **T1 GATE RESULT (2026-07-16): launch config = 128/2/4/512** — 93.4 ms/forward B=192 single-thread => ~25-33k projected sps at 16 threads (marginal pass); 192/3 and 256/4 fail hard (~10k / ~4k). T8's real 16-thread collect bench validates before deploy; fallback if it misses: GPU inference server on the remote 3060 or accept reduced sps through the kickstart phase. 256/4/8 remains the growth ceiling (needs GPU rollouts).
- **No deploy**: build/test/commit only; trainer swaps are a separate user-gated decision.
- `uv pip` (no pip binary); niced builds/benches; never touch training procs, checkpoints*/, league/registry.jsonl, remote.

## Obs v1 layout (single source of truth — schema/v1.toml mirrors this)

All positions/velocities normalized as v0 (pos·1/2300, vel·1/2300, ang_vel·1/5.5); mirroring = play-as-blue (negate x,y of all vectors for orange) exactly as v0's `mir`.

**Entity row: 26 floats** (`ENT_FEAT = 26`):
| idx | field |
|---|---|
| 0-4 | type one-hot: IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_PAD |
| 5-7 | pos (normalized, mirrored) |
| 8-10 | vel |
| 11-13 | ang_vel |
| 14-16 | forward (cars; zeros for ball/pad) |
| 17-19 | up (cars; zeros ball/pad) |
| 20 | boost 0..1 (cars) / pad big-timer 0..1 (pads: 0 = available, else respawn/10) / 0 ball |
| 21 | on_ground (cars) / pad is-available flag (pads) / 0 ball |
| 22 | has_flip (cars) |
| 23 | demoed (cars) / 0 |
| 24 | ball-pred horizon τ/2.0 (0 for real entities; 0.25/0.5/0.75/1.0 for the +0.5/1/1.5/2 s predicted-ball entities) |
| 25 | reserved 0 |

**Entity count: `MAX_ENT = 17`** = 1 ball + 6 cars + 6 big pads + 4 ball-pred. Absent cars → zero row + mask=1 (ignored). Order: self, mates (≤2), opps (≤3), ball, pads (fixed arena order), ball-pred (ascending τ).

**Query/self row: 64 floats** (`Q_FEAT = 64`): self entity's 26 + 34 small-pad timers (0..1, fixed arena order, mirrored pairing) + prev-action index history is NOT here — prev actions enter as **4 floats**: bias… — no: exact split = 26 (self entity) + 34 (pad timers) + 3 (scoreboard: score_diff/5 clamped, time_frac, overtime) + 1 (reserved) = 64. **Prev-5 actions** are appended to the *pooled* embedding, not the query row (5 × one-hot-92 is too wide): they enter as 5 embedded action vectors via the same action-embedding MLP as the head, summed with learned position weights — implemented inside the net, fed as `prev_actions [B,5] int64`.

**Net I/O contract (both Rust and Python):**
`forward(entities [B,17,26], mask [B,17] bool, query [B,64], prev_actions [B,5]) -> (logits [B,92], value [B,1])`

**Ball prediction (Rust):** per-arena persistent cloned car-less arena (`Arena::clone(false)` idiom); each env step, copy real ball state in, step 60 ticks (0.5 s) four times, reading the ball at each horizon. Measured cost ~121 µs full / much less warm — budget ≤10 µs/step amortized by re-predicting every step from scratch at 240 ticks ONLY when ball state deviates — keep it simple: full 240-tick repredict each env step (~121 µs per arena per step at tick_skip 8 = fine vs candle forward).

---

### Task T1: Candle transformer microbench (dims gate)

**Files:** Create `engine/examples/bench_transformer.rs`.
No test — the deliverable is numbers. Build a minimal pre-LN encoder forward in candle (embed 26→d, L blocks of {LN → MHA(h heads) → residual → LN → FF(d→ff→d) → residual}, query pooling = 1 cross-attention of the self token over outputs, logits via dot with 92 embedded actions) with random weights, batch sizes {48, 96, 192} × dims {128/2/4/512, 192/3/6/768, 256/4/8/1024}, seq 17, single thread (`ensure_single_thread_gemm`). Report µs/forward and projected collect sps (assume forward dominates; current MLP reference ~101k sps at 192 arenas remote).

- [ ] Write bench, `cargo run --release --example bench_transformer` (niced), record table into the plan-adjacent notes file `.superpowers/sdd/transformer-bench.md`.
- [ ] **Gate:** pick the largest config projecting ≥25k sps at 192 arenas (16 threads remote). If even 128/2 is below, escalate to the controller (options: GPU inference server, smaller d) — do NOT proceed silently.
- [ ] Commit: `bench: candle transformer inference microbench`.

### Task T2: schema v1 + action table v1.1

**Files:** Create `schema/v1.toml`; Modify `engine/src/actions.rs`, `engine/src/schema.rs`; Test `engine/src/actions.rs` tests + `engine/tests/schema_test.rs` (or existing schema tests' file).

**Interfaces:** `actions::make_lookup_table_v1() -> Vec<[f32;8]>` (92 rows, first 90 == v0 exactly), `actions::TABLE_SIZE_V1 = 92`; `Schema { version: 1, obs: ObsV1Meta { max_ent: 17, ent_feat: 26, q_feat: 64, prev_actions: 5 }, action_count: 92, ... }` (serde-default so v0 files still parse).

- [ ] RED: test `v1_table_appends_stalls_only` — first 90 rows bit-equal `make_lookup_table()`, rows 90/91 equal the two stall tuples; test schema v1 file parses with obs meta and v0 file still parses byte-identically.
- [ ] Implement; `schema/v1.toml` carries version=1, obs_size=0 (unused), entity dims, action_table="construct_92_v1", action_count=92, tick_skip=8, same normalization block.
- [ ] GREEN + commit: `feat: schema v1 + action table v1.1 (92, stall rows)`.

### Task T3: obs v1 Rust builder + ball-pred tracker

**Files:** Create `engine/src/obs_v1.rs`, `engine/src/ballpred.rs`; Modify `engine/src/lib.rs` (mod decls only); Tests inline `#[cfg(test)]` + `engine/tests/obs_v1_test.rs`.

**Interfaces:** `obs_v1::{MAX_ENT=17, ENT_FEAT=26, Q_FEAT=64, PREV_ACTIONS=5}`; `obs_v1::build(state: &GameState, car_idx: usize, pads: &PadState, pred: &[BallSnap;4], prev: &[i64;5], n: &Normalization, ents: &mut [f32], mask: &mut [bool], query: &mut [f32])`; `ballpred::Tracker::new()`, `Tracker::predict(&mut self, ball: &BallState) -> [BallSnap;4]` (cloned car-less arena, snapshots at +60/120/180/240 ticks). `PadState` = the engine's existing boost-pad timer access (find how episode.rs/obs.rs reads pads; if v0 doesn't read pads, add a `pads_from_arena(&mut Arena)` helper here).

- [ ] RED tests: (a) layout golden — hand-built 1v1 GameState → assert exact entity rows (self flags, ball row, pad rows, pred rows at τ slots) and mask (mates/opps absent → masked); (b) mirroring — orange car's obs equals blue-mirrored twin state (reuse v0's mirror-test pattern from obs.rs); (c) ballpred — stationary ball at kickoff predicts ~same pos at all horizons; ball moving +y at 2000 predicts increasing y; all finite.
- [ ] Implement. Determinism: Tracker per arena, owned by EpisodeArena (next task wires it); no global state.
- [ ] GREEN + commit: `feat: obs v1 entity builder + ball prediction tracker`.

### Task T4: candle entity policy (Rust inference)

**Files:** Create `engine/src/policy_v1.rs`; Test `engine/tests/policy_v1_test.rs` + golden fixture via `scripts/gen_policy_v1_fixture.py` (Create).

**Interfaces:** `policy_v1::EntityPolicy::new(weights: &HashMap<String, Tensor-like>) -> Result<Self,String>` consuming the EXACT state-dict names of the Python net (T5 defines them; the two tasks must agree — the fixture test enforces it); `forward(ents, mask, query, prev) -> (logits [B,92], values [B])`. Pre-LN encoder built from candle matmuls/softmax (no flash-attn; seq 17). `ensure_single_thread_gemm` reused.

- [ ] RED: golden test — `scripts/gen_policy_v1_fixture.py` builds the T5 Python net with fixed seed, saves weights + a batch of random (ents,mask,query,prev) + expected (logits,values) as npz fixture; Rust test loads fixture, asserts max-abs-diff < 1e-4.
- [ ] Implement (weight-name mapping table documented at top of file: `trunk.blocks.{i}.ln1.weight` etc.).
- [ ] GREEN + commit: `feat: candle entity-transformer inference (policy v1)`.

### Task T5: Python EntityPolicyNet

**Files:** Create `python/construct/learn/model_v1.py`; Test `tests/python/test_model_v1.py`.

**Interfaces:** `EntityPolicyNet(d_model, layers, heads, ff, action_table: np.ndarray [92,8], aux: bool=False)` with submodules named exactly: `embed` (26→d), `query_embed` (64→d), `act_embed` (MLP 8→d→32), `prev_embed_w` ([5] learned), `blocks.{i}.{ln1,attn,ln2,ff1,ff2}`, `pool` (query cross-attn), `policy_dot` (Linear d→32), `value_head` (d→1), optional `aux_reward` (d→3), `aux_recon` (d→17*26). `forward(ents,mask,query,prev) -> (logits, value)`; `act`/`evaluate` mirroring model.py's API so ppo.py plugs in unchanged. Action embeddings computed from the 92×8 table buffer each forward (cheap) — logits = act_emb @ player_emb (dot, no scaling; Nexto-verified).
- Param count printed; masked attention via additive −inf on masked keys.

- [ ] RED: tests — output shapes; mask invariance (changing a masked entity's features doesn't change logits); permutation equivariance of mates (swapping two mate rows leaves logits identical); prev-action sensitivity (changing prev changes logits).
- [ ] Implement + GREEN + commit: `feat: python entity policy net (pre-LN encoder, dot head)`.

### Task T6: engine v1 plumbing (EpisodeArena + Engine + collect)

**Files:** Modify `engine/src/episode.rs` (v1 obs branch + prev-action ring + Tracker owned per arena), `engine/src/engine.rs`, `engine/src/lib.rs` (Engine accepts schema v1: exposes `obs_mode`, entity buffer outputs; `set_weights` dispatches MlpPolicy vs EntityPolicy by schema version); Tests `engine/tests/engine_v1_test.rs`.

**Interfaces:** With `schema_path="schema/v1.toml"`, `Engine.collect(T)` returns dict with `ents [T,N,17,26]`, `mask [T,N,17]`, `query [T,N,64]`, `prev [T,N,5]` in place of `obs` (plus the unchanged rewards/flags/actions/logprobs/values/learner_agents keys). RenderSession gains the same branch. **v0 path untouched — assert byte-identity test still green.**

- [ ] RED: v1 collect smoke (2 arenas 1v1, random-weight net, 200 steps: shapes, finite, mask pattern correct for 1v1); v0 regression (existing byte-identity + all v0 tests).
- [ ] Implement (prev-action ring per agent, reset to zeros on episode reset; opponent slots also get v1 forward — league compatible).
- [ ] GREEN + commit: `feat: engine v1 entity collect path`.

### Task T7: kickstart distillation (trainer)

**Files:** Create `python/construct/learn/kickstart.py`; Modify `python/construct/learn/train.py`, `config.py`, `scripts/resume_train.py` (--kickstart-teacher CK --schema v1 flags); Test `tests/python/test_kickstart.py`.

**Interfaces:** `KickstartTeacher(ck_path)` — loads the frozen v0 MLP + builds v0 obs from... **decision locked:** teacher consumes the v0 94-float obs; engine v1 collect must ALSO emit `obs_v0 [T,N,94]` when `kickstart=true` (cheap add in T6's branch — gate by env flag `emit_v0_obs`; add to T6 scope). Loss: `L = L_PPO + λ_k · KL(teacher || student)` on collect states + `λ_v · MSE(V_student, V_teacher)`, λ_k linear-anneal `1.0 → 0` over `kickstart_steps` (config, default 500M), λ_v fixed 0.5 while λ_k>0. Teacher logits [B,90] pad to 92 with −inf for the 2 stall slots.
- [ ] RED: unit test with tiny random teacher/student — λ_k=1 gradient pulls student logits toward teacher (KL decreases over 50 steps on fixed batch); anneal schedule hits 0 at kickstart_steps; stall-slot padding never produces NaN.
- [ ] Implement + GREEN + commit: `feat: kickstart distillation (PPO + annealed teacher KL)`.

### Task T8: trainer integration + configs + local canary bench

**Files:** Modify `python/construct/learn/train.py` (v1 batch plumbing → GAE/PPO reshapes over dict-of-tensors), `configs/train_v1.toml` (Create: [net] dims from T1 gate, [kickstart] block, schema/v1.toml, reward_v3); Test `tests/python/test_train_v1.py` + `scripts/smoke_test.py --schema v1` mode (Modify).

- [ ] RED: v1 smoke — 2 iters tiny config end-to-end (collect→GAE→PPO update→checkpoint save/load roundtrip with v1 net); v0 smoke unchanged.
- [ ] Implement; checkpoint records schema_version=1 + net dims (resume-safe).
- [ ] Local canary bench (NO deploy): 60 s collect on this box at 48 arenas, record sps into `.superpowers/sdd/transformer-bench.md`.
- [ ] GREEN + commit: `feat: v1 trainer path + kickstart config`.

**Final:** whole-branch review (fable tier) → merge → controller presents deploy/kickstart-launch options to the user (ask-before-deploy).

## Post-plan notes
- BC (imitation back half) consumes T5's net + a replay→obs-v1 path — that path is a follow-up task on the replay crate (build obs v1 from shard rows), deliberately out of scope here.
- Aux heads ship config-gated OFF at launch (Lucy-SKG λ values unpublished; tune later).
- Deploy path (RLBot bot.py) needs the v1 obs builder in the deploy adapter — separate small task at next real-game deploy.
