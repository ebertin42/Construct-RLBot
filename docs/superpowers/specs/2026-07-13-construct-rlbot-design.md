# Construct — Rocket League Bot: Design Spec

**Date:** 2026-07-13
**Status:** Approved by user (section-by-section) — pending written-spec review
**Goal:** Push state of the art for RLGym-trained Rocket League bots across 1v1, 2v2, and 3v3. Local play only via RLBot v5 (offline-only by design; no EAC interaction).

## 1. Context and constraints

- **Hardware:** Local WSL2 box (16 threads, 19 GB RAM, RTX 4060 Laptop 8 GB) for development and medium runs; rented cloud machines for long campaigns.
- **Budget:** $2k+ available; collection is CPU-bound so cloud spend goes to high-core boxes (~$0.2–0.4/hr on vast.ai), not big GPUs. At Rust-engine throughput, 10B steps costs tens of dollars — the budget buys *many experiments*, which is the actual edge.
- **Data:** Full human-replay imitation pipeline (ballchasing.com).
- **Policy scope:** One shared entity-attention policy for all team sizes, optional per-mode finetunes late.
- **Distribution:** Weights stay private. Psyonix asked the community not to publicly release Nexto-caliber bots after the 2023 ranked-cheating incident. Local play and development are sanctioned uses of RLBot.

### Research grounding (key sources)

- **Seer thesis** (Walo, ETH): most complete published spec — replay-state resets (p=0.7), TrueSkill self-play protocol, schedules, and the negative result that BC-init alone washes out. <https://nevillewalo.ch/assets/docs/MA_Neville_Walo_Seer_RLRL.pdf>
- **Ripple** (Rolv): VPT-style IDM → BC on ~37k hours of SSL replays → PPO with decaying KL penalty toward the BC prior. The proven fix for BC washout. <https://github.com/Rolv-Arild/replay-pretraining>
- **Lucy-SKG** (arXiv:2305.15801): KRC multiplicative reward combination + auxiliary heads → ~5x sample efficiency vs Necto; beat Nexto 300:4 at equal steps.
- **Necto/Nexto** (Rolv): entity-attention (EARLPerceiver), 90-action lookup table, ControlsPredictorDot head, team-spirit blending, one net for 1s/2s/3s at ~GC1. <https://github.com/Rolv-Arild/Necto>
- **Ecosystem:** RLGym 2.0.1, rlgym-tools 2.6.5 (replay parsing, mutators, advanced rewards), rocketsim-rs, boxcars (Rust replay parser), RLViser, RLBot v5 (rc), rlgym-compat 2.3.6.
- **Negative result to respect:** Necto v3 (Tecko) cancelled — naive scaling plateaus. Marginal gains live in replay resets, IL priors held via KL, opponent leagues, and raw throughput.

## 2. Architecture

Approach: **Rust engine + Python brain.** Rust owns the CPU-bound hot loop (sim stepping, obs building, reward computation, mutators); Python owns everything researchy (PPO, BC/IDM, analysis, league logic). Bridged in-process via PyO3/maturin with zero-copy numpy arrays; split into separate processes only when multi-box distribution demands it.

```
┌─────────────── Rust: construct-engine ───────────────┐
│ N worker threads × M arenas (rocketsim-rs)           │
│ sim stepping (tick_skip 8) · obs building ·          │
│ reward computation (TOML config) · state mutators ·  │
│ action lookup table · ball prediction                │
│ ⇄ zero-copy numpy: obs batches out, actions in       │
└──────────────────────────────────────────────────────┘
   Python: construct-learn   — batched GPU inference, PPO (+KL, aux heads), checkpoints, WandB
   Python: construct-data    — ballchasing downloader, boxcars parsing, IDM, BC
   Python: construct-league  — checkpoint pool, TrueSkill, opponent sampling, eval matches
   Python: construct-deploy  — RLBot v5 bot (Windows), rlgym-compat, TorchScript export
```

**Versioned interface schema (single source of truth):** obs layout, action table, and reward config live in one TOML/JSON spec consumed by both Rust and Python (runtime-checked at load on both sides). Every checkpoint records its schema version. This prevents the classic sim/deploy obs-mismatch bug and keeps historical checkpoints loadable.

**Repo layout:** Cargo workspace under `engine/`; Python packages `learn/`, `data/`, `league/`, `deploy/`; one uv-managed venv (Python 3.11 — the ecosystem sweet spot); maturin builds the engine as an importable Python module.

## 3. Policy network, observations, actions

**Observations — entity-based, player-count-invariant** (enables one policy for all modes):

- Entities: ball, cars (masked padding to 6), 6 big boost pads. Per entity: position, velocity, angular velocity, forward/up vectors, boost, on-ground / has-flip / demoed flags, team-relative flag. Normalized; relative-to-self plus absolute.
- Self features: own physics, boost, flip state, 34 small-pad timers (compact vector), previous k=5 actions stacked (Lucy-SKG: compensates for no recurrence), scoreboard (score diff, time remaining, overtime).
- **Ball-prediction entities** (Ripple's trick): predicted ball positions at +0.5/1/1.5/2 s from the rocketsim ball predictor, computed in Rust.
- Mirroring: always "play as blue," plus left-right augmentation.

**Network:** pre-LN transformer encoder — start at 256 dim, 4 layers, 8 heads (~3M params); profile and grow if the GPU idles. Self-entity query pooling into shared trunk; policy and value heads. **No LSTM** — pad timers are fed directly, and feedforward keeps throughput and BC simple.

**Policy head:** action-embedding dot product (ControlsPredictorDot) — each discrete action embedded via small MLP; logits = dot(action embedding, player embedding). Scales to larger tables at no head cost.

**Auxiliary heads** (config-gated, on by default per Lucy-SKG ablations): reward-prediction head and state-reconstruction head on the trunk.

**Actions:** curated lookup table of ~110–130 entries — the standard 90 plus curated extensions for flip resets, wavedashes, and stalls. Table is part of the shared schema. Tick skip 8 (15 Hz decisions). Tick skip 4 is a late-phase experiment, supported by schema but out of scope for P0–P2.

## 4. Rewards and curriculum

**Reward system** — implemented in Rust, driven by TOML config (weight/stage changes need no recompile):

- Component library: Lucy-SKG KRC combos (Offensive Potential, Distance-weighted Alignment, Touch Ball-to-Goal Acceleration), Nexto components (touch-height, sqrt-scaled boost gain/loss, demo, flip-reset), guide staples (speed-to-ball, velocity-ball-to-goal, air, save-boost, face-ball).
- All components bounded to [-1, 1]; weights set magnitude.
- **Selective zero-sum wrapping** on contested quantities only (goals, boost, demos) — never on movement shaping.
- **Team spirit** blending `r = (1−τ)·own + τ·team_mean − opp_mean`, τ annealed ~0.2 → 0.6+ across training.
- **Four staged config generations,** hot-swapped at checkpoints:
  1. *Foundations:* touch/speed-to-ball/face-ball/air heavy; goal ≈ 20 (never 100 — kills exploration).
  2. *Shot-making:* velocity-change-scaled touches, KRC offensive potential, boost economy.
  3. *Competition:* strip most shaping; goal/win dominant; concede = −goal·(1−aggression_bias), bias ≈ 0.2; mechanic bonuses (flip reset, wavedash, aerial distance).
  4. *Sparse + KL:* near-sparse goal/win; KL-to-human-prior carries style regularization.
- Telemetry: per-component running stats to WandB; **reward report over the replay corpus** (assert the reward function ranks SSL play highly).

**Curriculum (weighted state mutators):**

- Base mix (Seer-proven): **replay-state resets 0.7**, kickoff 0.1, scenario packs 0.15 (aerial/wall/goalie/training-pack states), random physics 0.05.
- **Variable team size per episode** (1v1/2v2/3v3 sampled; weights configurable — more 1v1 early for signal density).
- Domain randomization: all 6 hitboxes, random scoreboards, occasional physics jitter.
- Hyperparameter schedule: batch 50k → 300k, LR 2e-4 → 0.8e-4, entropy 0.01 → 0.005, γ horizon 10 s → 20 s. Schedule-driven and checkpoint-resumable.

## 5. Imitation pipeline

1. **Acquire:** ballchasing.com API downloader — rank-filtered (GC3/SSL + RLCS), all playlists, dedup, resumable. Target 100k+ replays. Store raw + parsed.
2. **Parse:** boxcars (Rust) → normalized frame sequences (physics per tick, jump/dodge events, boost) → `.npy` shards. Same pipeline feeds ReplayMutator reset states.
3. **IDM (VPT-style):** replays lack pitch/yaw/roll inputs. Train an inverse dynamics model on bot-generated sim data (ground-truth actions known): ±20-frame physics window → action from our lookup table. Validate against replay jump/dodge events (stored in replays — free ground truth). Label the corpus.
4. **BC pretrain:** train the exact policy net on labeled SSL data via the Rust replay→obs path (guarantees train/deploy consistency). Inverse-action-frequency-weighted cross-entropy. Value head warm-started with discounted-return regression. Winners-side-only filtering for finetune-grade data.
5. **RL with prior:** PPO + **KL penalty toward frozen BC policy, decayed over training** (Ripple's fix for Seer's washout). Config-gated for ablations.
6. **Ablation discipline:** pure-RL vs BC-init vs BC-init+KL as separate WandB groups. Measure, don't assume.

## 6. Self-play league and evaluation

- **Rollout opponents:** ~80% latest self, ~20% past checkpoints sampled PFSP-lite: TrueSkill-gated (μ > μ_current − 10), weighted toward beatable-but-hard.
- **Pool seeds:** BC policy (human-style anchor), public Nexto weights, RLBotPack scripted anchors (sim-side re-implementation where feasible, else eval-only).
- **Exploiter slot (late phase, config-gated):** Minimax-Exploiter-style agent trained only vs current main; main trains against what it finds.
- **Ratings:** continuous TrueSkill ladder (Seer protocol — inherit μ, ~20 max-information matches, freeze into league). Checkpoint promotion by win-rate A/B, never by reward.
- **Per-mode ratings** (1v1/2v2/3v3) tracked separately for the shared policy.
- **Behavior dashboard:** aerial touch rate, ball height on touch, boost economy, demo rate, kickoff win %, possession — regression detection for reward changes.
- **Real-game validation:** periodic RLBot v5 deployment; scripted matches vs RLBotPack bots; human eyeball games.
- **Registry:** SQLite + files — weights, config generation, schema version, rating history. Any checkpoint replayable.

## 7. Phases

| Phase | Scope | Exit criterion |
|---|---|---|
| **P0 — Walking skeleton** | Rust engine (rocketsim-rs, basic obs/reward/table) + PyO3 bridge + minimal PPO + RLViser + RLBot v5 deploy of one checkpoint | Ball-chasing bot visible in the real game; loop proven end-to-end |
| **P1 — Training program** | Full obs/net, reward library + stages, curriculum mutators, league + TrueSkill, WandB | Overnight local runs with measurably climbing ratings |
| **P2 — Imitation** | Downloader, parsing, ReplayMutator states, IDM, BC, KL-PPO | BC-init+KL beats pure-RL at equal steps |
| **P3 — Scale campaign** | Cloud rollout boxes, multi-box collection, long campaigns, exploiters, tick-skip-4 experiments, per-mode finetunes | Beats Nexto, then hunts pack top tier |

## 8. Infrastructure and deployment

- **Local:** dev, ablations, P0–P2 runs. WSL2 trains headless; RLViser runs Windows-side via UDP when needed.
- **Cloud:** vast.ai high-core CPU boxes as rollout workers; learner stays on the GPU box. Everything resumable (checkpoint + config generation + schema version) so spot boxes can die freely.
- **Deployment (Windows):** RLBot v5 bot; rlgym-compat GameState → same versioned obs schema (checked at load); TorchScript export; CPU inference within the 8-tick budget; engine trains with `rlbot_delay=true` so sim timing matches RLBot's 1-tick action delay. Match configs for 1v1/2v2/3v3 vs Psyonix bots, pack bots, or the user.

## 9. Testing

- **Rust:** unit tests per obs/reward component against known-state fixtures; sim determinism tests; schema round-trip tests.
- **Python:** PPO math tests (GAE, KL, ratio clipping on toy envs); BC/IDM overfit-one-batch sanity checks.
- **Sim/deploy parity test (killer-bug preventer):** same recorded GameState through the training obs builder and the deploy obs builder must produce bit-identical output.
- **Integration:** 1k-step training smoke test runnable as a single script.

## 10. Out of scope (explicit)

- Public release of trained weights (Psyonix request).
- Online/ranked play of any kind (RLBot is offline-only; this project targets local matches).
- Pixel-based observations, model-based RL, full AlphaStar league with concurrent exploiter populations (single exploiter slot only — full league is a possible P4).
- Tick-skip-4 and per-mode finetunes before P3.

## P0 Results (2026-07-14)

| Metric | Random baseline | 107M steps | 150M steps |
|---|---|---|---|
| touches/min/agent | 0.00 | 15.16 | 31.97 |
| mean dist-to-ball | 3769 uu | 1294 uu | 1288 uu |

- Training throughput: ~48-58k policy-steps/sec (64 arenas, 14 worker threads, RTX 4060 Laptop learner+inference; ~20k when thermally throttled). Raw engine bench without NN: 87k env-steps/sec.
- ep_reward: 3.5 (5M) → 10-12 (40M) → 15.8 (150M); goals scored regularly in self-play.
- Eval gates (≥3x baseline touches, <half baseline dist): passed at 107M.
- Checkpoint/resume exercised in production (pause + exact resume at 114.7M).
- RLViser streaming verified (WSLg local; Windows-host streaming via CONSTRUCT_VISER_ADDR).
- Remaining P0 exit item: bot visible in real Rocket League via RLBot v5 (Windows-side, user-assisted).
