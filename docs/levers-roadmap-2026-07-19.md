# Levers Roadmap — 2026-07-19 (post-research synthesis)

Sources: internal spec/plan remnants inventory + deep-research sweep (11 verified
findings, 3-0 adversarial votes except where noted; Seer thesis, Lucy-SKG
arXiv:2305.15801, AlphaStar Nature, Ubisoft Minimax Exploiter, COLT-2025
imitation lower bound, Necto-lineage repos).

## The reframe that matters tonight

**Seer's BC prior (820,958 Diamond+ replays, SSL-finetuned) also could not play
in closed loop**, and their human-prior *init* run collapsed to the random-init
curve — they shipped it with NO KL regularization. COLT-2025 proves smooth
imitators suffer exponentially-compounding closed-loop error regardless of
held-out accuracy. Conclusions for us:

1. Our B6 closed-loop failures (0-98 vs 520M) are the EXPECTED vanilla-BC
   outcome — data poisoning made it worse but was never the whole story.
2. Our KL-prior design is precisely the fix class the literature prescribes
   (frozen prior as anchor ≠ prior as init). The anchor consumes conditional
   action distributions at PPO's OWN states — compounding error doesn't apply.
   Risk that remains: prior's conditionals off its training distribution may
   be poor → mitigate with modest λ_p + the reset-pool lever below (which
   moves PPO's state distribution TOWARD the prior's).
3. **B6 gate revised**: deploy K4 when the probe shows good state-conditional
   competence (prev-zeroed top-1 ≥ 0.40, copy-rate near human 0.70, sane
   top-k) and closed-loop shows any structure above copycat baseline — match
   wins NOT required. λ_p 0.05, eval-watched, restart-without-flag as
   kill-switch.

## Ranked levers (impact ÷ cost, our 2-box scale)

1. **Replay-state resets 0.7 mix** (spec §4, INTERNAL, mostly built): reset
   pools already on disk (16 states × 67.5k replays, v5). Engine task: sample
   reset_pool_v5.jsonl in EpisodeArena resets (0.7 replay / 0.1 kickoff /
   0.15 scenario / 0.05 random). Seer-proven; synergizes with the KL-prior
   (state distributions align). NEXT ENGINE TASK.
2. **Aux reward-prediction head** (Lucy-SKG, verified: halved steps-to-plateau
   100M vs 200M): 3-class immediate-reward head branched from shared layers
   (skip their autoencoder aux — no gain). Net has aux=False plumbing ready.
3. **Reward v4 design pass** (Seer+Lucy-SKG verified): zero-sum by
   opponent-subtraction (kills goal-trading arithmetically — better than our
   -12 asymmetry which caused avoidance), annealed goal weight 1.25→1.45,
   multiplicative touch decay (×0.95, floor 0.1), horizon anneal 10s→20s,
   KRC sign-gated geometric means for correlated shaping (KRC-Necto beat
   Necto 300-220 at matched compute).
4. **Seer 80/20 pool tuning** (near-zero cost): we run league 20% opponents —
   align gating (TrueSkill μ > μ_new − 10, μ-weighted sampling). Mostly config.
5. **Obs audit for throughput** (Lucy-SKG: 2x FPS from slimming to 8 attended
   objects + prev-5 stacking — we already stack prev-5; our 17-slot entity obs
   could drop small-pad/timer detail). Profile first.
6. **Minimax exploiter** (Ubisoft, fits 2-GPU): main + one exploiter whose
   reward adds −αγ·max_a Q_main(s′,a) from the frozen main's value net;
   85% win-rate convergence gates; proven on one game machine + two GPUs.
   Medium cost; after KL-PPO era stabilizes.
7. **Non-causal IDM audit of B2 labels** (community-standard VPT-style over
   past+future states): our analytic projection may carry label noise the
   probe can't see. Cheap audit: train small IDM, measure disagreement with
   B2 labels; escalate to relabel only if disagreement high.
8. **BC that can actually play** (only if we need a standalone human bot):
   action chunking + noise-injected/exploratory data + stochastic heads
   (COLT-2025-aligned escapes). NOT needed for the anchor use-case.
9. **P3-lite async collect** (laptop rollout worker → remote learner): Seer's
   author names async distributed rollout as the missing scale lever; ~+60%
   steps/s here. After league/exploiter work.

## Scale calibration (verified)

Seer: 1e10 steps ≈ Platinum I. Lucy-SKG: design levers ≈ 5x sample
efficiency (beat 1B-step Necto from 200M). At our ~8k steps/s (1e10 ≈ 2
weeks), DESIGN DOMINATES SCALE — levers 1-5 before any compute expansion.

## Internal remnants not in the ranked list

Value-head warm-start (B5 deferral), rlbot_delay 1-tick queue (live-game
parity), mechanic bonuses (reward stage-3), WandB, vhdx compaction, per-mode
finetunes / tick-skip-4 (P3).
