# Jump / Flip / Boost Reward Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the policy a reachable aerial payout and a reason to take boost pads, then fork a v10 run to test it without risking the live v9 run.

**Architecture:** Three new terms in `engine/src/reward.rs` — one reward (`boost_pickup`) and two instruments (`boost_gained`, `flip_events`) — plus a new reward TOML that lowers the aerial payout ramp floor from 400uu to 150uu while holding the measurement floor at 400uu. Ship as a wheel to the remote box and start a second trainer against a new checkpoint directory. The live v9 run is never touched.

**Tech Stack:** Rust (rocketsim_rs 0.37.0, maturin/PyO3), Python 3.12, TOML config, remote box at `elliot@192.168.86.117`.

## Global Constraints

- **NEVER run `maturin develop`.** It overwrites `python/construct/_engine.abi3.so`, sha `deab5d3b988d62d332200c51015269427229a01986f81876e905a91308eaffd9` — the local gate instrument every gate and bench result in this session was scored on. Build with `maturin build -o <dir outside the repo>` only.
- **Never restart the v9 run (run A).** It is at its fastest measured improvement rate. Its process holds the old `.so` in memory and is unaffected by installing a new wheel on the remote.
- `cfg.boost_pickup` MUST carry `#[serde(default)]` so every existing reward TOML stays reward-identical.
- New terms are APPENDED at indices 20, 21, 22. Never renumber existing terms — `TERM_NAMES` is positional and checkpoints/telemetry depend on the order.
- Python reads reward terms by NAME from a dict (`train.py:1550` `reward_terms()`), so appending is safe; do not assume positional access.
- `aerial_meas_z_lo` stays 400.0 in the new config. Moving it would move the instrument that scores the experiment.
- Remote runs an INSTALLED package at `.venv/lib/python3.12/site-packages/construct/`, NOT the repo. Editing repo files on the remote changes nothing until a wheel is installed.

---

### Task 1: Add the three terms to the engine

**Files:**
- Modify: `engine/src/reward.rs` (N_TERMS at :127, term consts at :128-184, `TERM_NAMES` at :187, `RewardConfig` struct at :6-114, `compute_terms` at :395)
- Test: `engine/src/reward.rs` (inline `#[cfg(test)] mod tests`, existing pattern at :1561+)

**Interfaces:**
- Consumes: `compute_terms(prev: &GameState, cur: &GameState, car_idx: usize, scored: Option<Team>, cfg: &RewardConfig, terms: &mut [f64; N_TERMS]) -> f32`
- Produces: `T_BOOST_PICKUP: usize = 20`, `T_BOOST_GAINED: usize = 21`, `T_FLIP_EVENTS: usize = 22`, `N_TERMS = 23`, `RewardConfig.boost_pickup: f32`

- [ ] **Step 1: Write the failing tests**

Add to the existing `mod tests` in `engine/src/reward.rs`:

```rust
    /// A pad pickup is a POSITIVE boost delta, scaled by SQRT so it pays only
    /// in proportion to how much the boost was NEEDED.
    #[test]
    fn boost_pickup_pays_on_sqrt_of_amount_gained() {
        let (mut prev, mut cur) = synth();
        prev.cars[0].state.boost = 0.0;
        cur.cars[0].state.boost = 100.0;
        let mut cfg = cfg_zero();
        cfg.boost_pickup = 0.2;
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert!((t[T_BOOST_PICKUP] - 0.2).abs() < 1e-6,
                "big pad from empty pays the full coefficient; got {}", t[T_BOOST_PICKUP]);
        assert!((t[T_BOOST_GAINED] - 100.0).abs() < 1e-6,
                "instrument counts RAW units, not the sqrt-shaped reward");

        // small pad from empty: (sqrt(12) - 0)/10 = 0.3464
        let mut t2 = [0.0f64; N_TERMS];
        cur.cars[0].state.boost = 12.0;
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t2);
        assert!((t2[T_BOOST_PICKUP] - 0.2 * 0.34641).abs() < 1e-5,
                "small pad from empty; got {}", t2[T_BOOST_PICKUP]);
    }

    /// The whole point of sqrt: a pad taken when nearly full must pay almost
    /// nothing, and a pad taken when FULL must pay exactly nothing. A linear
    /// `gained/100` gets the full case right but prices 0->20 and 80->100 the
    /// same, which is wrong -- the first is desperate, the second worthless.
    #[test]
    fn a_pad_pays_only_to_the_extent_the_boost_was_needed() {
        let mut cfg = cfg_zero();
        cfg.boost_pickup = 0.2;

        let pay = |b0: f32, b1: f32| {
            let (mut prev, mut cur) = synth();
            prev.cars[0].state.boost = b0;
            cur.cars[0].state.boost = b1;
            let mut t = [0.0f64; N_TERMS];
            compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
            t[T_BOOST_PICKUP]
        };

        // ALREADY FULL: driving over a pad gains nothing, so it pays nothing.
        assert_eq!(pay(100.0, 100.0), 0.0, "a full car gains nothing from a pad");

        // NEARLY full: a big pad is worth ~1/16 of the same pad taken empty.
        let empty_big = pay(0.0, 100.0);
        let nearly_full_big = pay(88.0, 100.0);
        assert!(nearly_full_big > 0.0, "12 units gained is still worth something");
        assert!(nearly_full_big < empty_big / 10.0,
                "a pad at 88 boost must pay <1/10 of the same pad at 0: {nearly_full_big} vs {empty_big}");

        // A SMALL pad while empty must beat a BIG pad while nearly full --
        // this is the ordering a linear scaling gets backwards.
        assert!(pay(0.0, 12.0) > nearly_full_big,
                "need beats quantity: small-pad-when-empty must outpay big-pad-when-full");
    }

    /// Spending boost must NOT be penalised. This is the whole reason the term
    /// is an EVENT and not a potential: a potential on boost amount yields a
    /// hoarder, the same gamma-drag trap that left air_setup net negative.
    #[test]
    fn spending_boost_is_never_penalised() {
        let (mut prev, mut cur) = synth();
        prev.cars[0].state.boost = 100.0;
        cur.cars[0].state.boost = 40.0;
        let mut cfg = cfg_zero();
        cfg.boost_pickup = 0.2;
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_BOOST_PICKUP], 0.0, "burning boost must cost nothing");
        assert_eq!(t[T_BOOST_GAINED], 0.0, "and must not register as a pickup");
    }

    /// A tape that does not set boost_pickup is bit-identical to today.
    #[test]
    fn boost_pickup_defaults_to_zero_reward() {
        let (mut prev, mut cur) = synth();
        prev.cars[0].state.boost = 0.0;
        cur.cars[0].state.boost = 100.0;
        let cfg = cfg_zero(); // boost_pickup untouched -> 0.0
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_BOOST_PICKUP], 0.0, "unset coefficient pays nothing");
        assert!((t[T_BOOST_GAINED] - 100.0).abs() < 1e-6,
                "but the INSTRUMENT still records, so v9 telemetry is comparable");
    }

    /// The flip instrument fires on the has_flipped rising edge -- that is the
    /// dodge, not the first jump.
    #[test]
    fn flip_events_counts_the_dodge_rising_edge() {
        let (mut prev, mut cur) = synth();
        prev.cars[0].state.has_flipped = false;
        cur.cars[0].state.has_flipped = true;
        let cfg = cfg_zero();
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_FLIP_EVENTS], 1.0, "rising edge is one dodge");

        // Still flipped on the next tick is the SAME dodge, not a new one.
        let mut t2 = [0.0f64; N_TERMS];
        prev.cars[0].state.has_flipped = true;
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t2);
        assert_eq!(t2[T_FLIP_EVENTS], 0.0, "held-high must not double count");
    }

    /// TERM_NAMES is positional and consumed by name on the Python side.
    #[test]
    fn term_names_covers_every_index_and_is_unique() {
        assert_eq!(TERM_NAMES.len(), N_TERMS);
        assert_eq!(TERM_NAMES[T_BOOST_PICKUP], "boost_pickup");
        assert_eq!(TERM_NAMES[T_BOOST_GAINED], "boost_gained");
        assert_eq!(TERM_NAMES[T_FLIP_EVENTS], "flip_events");
        let mut seen = std::collections::HashSet::new();
        for n in TERM_NAMES { assert!(seen.insert(n), "duplicate term name {n}"); }
    }
```

If `synth()` and `cfg_zero()` do not already exist as test helpers with those exact names, find the equivalents already used by `aerial_of` (near `engine/src/reward.rs:1555`) and use those instead — do not write new fixtures.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/steamo/Construct-RLBot/engine && cargo test --lib boost_pickup 2>&1 | tail -20
```

Expected: FAIL — `cannot find value T_BOOST_PICKUP in this scope`.

- [ ] **Step 3: Add the constants and names**

In `engine/src/reward.rs`, change `pub const N_TERMS: usize = 20;` (line 127) to `23`, and append after `T_AIR_TOUCH_Z5` (line 184):

```rust
/// Reward: boost collected, scaled by amount gained so a big pad (100) pays
/// ~8x a small pad (12). An EVENT, not a potential -- a potential on boost
/// AMOUNT penalises SPENDING it and yields a hoarder, which is the same
/// gamma-drag that left `air_setup` net negative for 2B steps.
pub const T_BOOST_PICKUP: usize = 20;
/// Instrument: raw boost units gained, independent of `cfg.boost_pickup`, so
/// retuning the payout does not move the measurement. Same payout/instrument
/// split as `aerial_z_lo` vs `aerial_meas_z_lo`.
pub const T_BOOST_GAINED: usize = 21;
/// Instrument: dodge fires, counted on the `has_flipped` RISING EDGE. Flips
/// were previously unmeasurable -- the question "does the bot flip?" could
/// only be answered by watching the viewer.
pub const T_FLIP_EVENTS: usize = 22;
```

Append to `TERM_NAMES` (after `"air_touch_z_ge1200",` at line 206):

```rust
    "boost_pickup",
    "boost_gained",
    "flip_events",
```

- [ ] **Step 4: Add the config field**

In the `RewardConfig` struct (after `air_setup_z_hi` around line 114):

```rust
    /// Reward per boost pad collected, scaled by amount gained (`gained/100`).
    /// 0.2 means a big pad is worth ~4 decisions of forgone ball approach at
    /// `vel_to_ball = 0.05`, i.e. roughly a 0.27s detour -- a pad already on
    /// the path is always worth taking, a real detour never is.
    #[serde(default)]
    pub boost_pickup: f32,
```

- [ ] **Step 5: Implement the accumulation**

In `compute_terms`, immediately after `terms[T_AGENT_STEPS] += 1.0;` (line ~405):

```rust
    // Boost pickup + flip instrument. Both read `prev`'s matching car, so they
    // sit here rather than in the `touched` block -- neither depends on contact.
    {
        let was = &prev.cars[car_idx];
        let gained = me.state.boost - was.state.boost;
        if gained > 0.0 {
            terms[T_BOOST_GAINED] += gained as f64;
            if cfg.boost_pickup != 0.0 {
                // SQRT, not linear: boost has diminishing marginal value, so a
                // pad must pay in proportion to how much it was NEEDED. A pad
                // taken at 88 boost pays ~1/16 of the same pad taken empty, and
                // a full car gains 0 so it pays 0. Linear `gained/100` prices
                // 0->20 and 80->100 identically, which is wrong.
                // /10 normalises sqrt(100) = 10 to a [0,1] range.
                let g = (me.state.boost.sqrt() - was.state.boost.sqrt()) / 10.0;
                if g > 0.0 {
                    let b = cfg.boost_pickup * g;
                    r += b;
                    terms[T_BOOST_PICKUP] += b as f64;
                }
            }
        }
        // RISING edge only: `has_flipped` stays true for the rest of the air
        // time, so testing the level would count one dodge every tick.
        if me.state.has_flipped && !was.state.has_flipped {
            terms[T_FLIP_EVENTS] += 1.0;
        }
    }
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd /home/steamo/Construct-RLBot/engine && cargo test --lib 2>&1 | tail -15
```

Expected: PASS, and the pre-existing suite still green (146 lib tests before this change).

- [ ] **Step 7: Commit**

```bash
cd /home/steamo/Construct-RLBot
git add engine/src/reward.rs
git commit -m "feat(reward): boost_pickup term + flip/boost instruments

The policy has no reason to take boost pads: no term exists, while
vel_to_ball pays for going at the ball, so a detour is a small loss and a
free pad is worth exactly zero. Adds an EVENT reward on boost gained --
not a potential, which would penalise SPENDING boost and yield a hoarder.

flip_events makes dodges measurable for the first time; boost_gained is
coefficient-free so retuning the payout does not move the instrument."
```

---

### Task 2: Write the v10 reward config

**Files:**
- Create: `configs/reward_v10_jumpboost.toml`
- Test: `engine/src/reward.rs` (validation test)

**Interfaces:**
- Consumes: `RewardConfig.boost_pickup` and `validate()` from Task 1
- Produces: `configs/reward_v10_jumpboost.toml`, loadable by `resume_train.py --config`

- [ ] **Step 1: Write the failing validation test**

```rust
    /// The v10 tape must lower the PAYOUT floor while leaving the INSTRUMENT
    /// floor at 400 -- otherwise run A and run B are not comparable.
    #[test]
    fn v10_tape_moves_the_payout_floor_but_not_the_instrument() {
        let toml = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/reward_v10_jumpboost.toml"))
            .expect("configs/reward_v10_jumpboost.toml must exist");
        let cfg: RewardConfig = toml::from_str(&toml).expect("v10 tape must parse");
        cfg.validate().expect("v10 tape must validate");
        assert_eq!(cfg.aerial_z_lo, 150.0, "payout floor moves to where touches are");
        assert_eq!(cfg.aerial_meas_z_lo, 400.0, "instrument must NOT move");
        assert_eq!(cfg.aerial_z_hi, 1400.0, "ramp top unchanged");
        assert_eq!(cfg.boost_pickup, 0.2);
        assert!(cfg.aerial_z_lo > BALL_RADIUS,
                "a floor below the resting ball centre would pay for a ball on the floor");
    }
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /home/steamo/Construct-RLBot/engine && cargo test --lib v10_tape 2>&1 | tail -10
```

Expected: FAIL — file does not exist.

- [ ] **Step 3: Create the config**

Copy the live v9 tape and change exactly three things:

```bash
cd /home/steamo/Construct-RLBot
scp elliot@192.168.86.117:~/construct/configs/reward_v9_aerial.toml configs/reward_v10_jumpboost.toml
```

Then edit `configs/reward_v10_jumpboost.toml`:

- `aerial_z_lo = 400.0` -> `aerial_z_lo = 150.0`
- leave `aerial_meas_z_lo = 400.0` EXACTLY as is
- append at the end of the file:

```toml
# Reward per boost pad collected, scaled by amount gained (gained/100).
# 0.2 makes a big pad worth ~4 decisions of forgone ball approach at
# vel_to_ball = 0.05 (~0.27s detour): a pad already on the path is always
# worth taking, a real detour never is. 0.5 would buy ~0.67s and distort play.
boost_pickup = 0.2
```

Add a header comment at the top of the file recording why the floor moved:

```toml
# reward v10 = v9 + a REACHABLE aerial payout + boost pads.
#
# v9's aerial_touch ramp started at 400uu on ball-z. Measured over 122 windows
# / 25,016 airborne touches, the ball-z distribution at touch is 69% <150uu,
# 29% 150-300uu, 1.67% >300uu -- and that top figure is FLAT across 2B steps.
# The payout therefore never fired once, while jumping carried a small constant
# cost, so the policy correctly drove p_jump to 2.5e-3 (~123x below uniform).
# Once jumping collapsed, neither aerials nor flips could be discovered.
#
# height_ramp is graded, not a step, so lowering the floor to 150 RAISES payout
# for every touch above it without a cliff: 300uu 0 -> 12%, 800uu 40% -> 52%.
# aerial_meas_z_lo stays at 400 so the instrument that scores the experiment
# does not move with the payout.
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/steamo/Construct-RLBot/engine && cargo test --lib v10_tape 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Run the whole suite**

```bash
cd /home/steamo/Construct-RLBot/engine && cargo test 2>&1 | tail -8
```

Expected: all green. `test_selfplay_collect_is_deterministic_in_a_fresh_process` is a known ASLR flake under full-suite runs — if only that fails, re-run it in isolation to confirm.

- [ ] **Step 6: Commit**

```bash
cd /home/steamo/Construct-RLBot
git add configs/reward_v10_jumpboost.toml engine/src/reward.rs
git commit -m "feat(config): reward v10, aerial payout floor 400 -> 150uu

The instrument floor (aerial_meas_z_lo) deliberately stays at 400 so
air_gate_frac keeps scoring run A and run B on the same bar."
```

---

### Task 3: Build and ship the wheel

**Files:**
- Create: `~/construct-ship/` (outside the repo)
- Modify: nothing in the repo

**Interfaces:**
- Consumes: the built engine from Tasks 1-2
- Produces: an installed wheel on `elliot@192.168.86.117`

- [ ] **Step 1: Record the local gate .so hash BEFORE building**

```bash
cd /home/steamo/Construct-RLBot
sha256sum python/construct/_engine.abi3.so
```

Expected: `deab5d3b988d62d332200c51015269427229a01986f81876e905a91308eaffd9`. Write it down. If it already differs, STOP — the instrument has drifted and every bench number this session needs re-checking.

- [ ] **Step 2: Build the wheel OUTSIDE the repo**

```bash
mkdir -p ~/construct-ship
cd /home/steamo/Construct-RLBot
.venv/bin/maturin build --release -m engine/Cargo.toml -o ~/construct-ship 2>&1 | tail -5
```

**NEVER `maturin develop`.** It installs in-place and overwrites the gate `.so`.

- [ ] **Step 3: Verify the local .so is UNCHANGED**

```bash
cd /home/steamo/Construct-RLBot
sha256sum python/construct/_engine.abi3.so
```

Expected: still `deab5d3b988d62d3...`. If it changed, restore it from `~/construct-measure/gate_so_backup/` before going further — every gate and bench result in this session was scored on that exact binary.

- [ ] **Step 4: Ship and install on the remote**

```bash
WHEEL=$(ls -t ~/construct-ship/*.whl | head -1)
echo "shipping $WHEEL"
scp "$WHEEL" elliot@192.168.86.117:/tmp/
ssh elliot@192.168.86.117 "cd ~/construct && .venv/bin/pip install --force-reinstall --no-deps /tmp/$(basename $WHEEL)"
```

- [ ] **Step 5: Verify the new terms are live on the remote AND run A is untouched**

```bash
ssh elliot@192.168.86.117 'cd ~/construct && .venv/bin/python -c "
from construct import _engine
print(\"terms:\", _engine.n_terms() if hasattr(_engine,\"n_terms\") else \"n/a\")
"; echo "run A trainers (MUST still be 1): $(pgrep -cf "[t]rain.py")"'
```

Expected: run A still reports exactly 1 trainer and has not restarted. If `n_terms()` is not exposed, verify instead by starting run B in Task 4 and checking `boost_gained` appears in its log.

- [ ] **Step 6: Commit (no repo change; record the ship)**

No commit — nothing in the repo changed. Note the wheel filename in the task log.

---

### Task 4: Fork run B

**Files:**
- Create on remote: `~/construct/checkpoints_v10/`
- Modify: nothing on run A

**Interfaces:**
- Consumes: the installed wheel from Task 3, `configs/reward_v10_jumpboost.toml` from Task 2
- Produces: a second trainer writing to `checkpoints_v10/`

- [ ] **Step 1: Ship the config and pick the fork point**

```bash
cd /home/steamo/Construct-RLBot
scp configs/reward_v10_jumpboost.toml elliot@192.168.86.117:~/construct/configs/
ssh elliot@192.168.86.117 'ls -t ~/construct/checkpoints_v9/*.pt | head -1'
```

Record the checkpoint path — that is the common ancestor and must be named in the run B launch.

- [ ] **Step 2: Confirm run A is healthy and capture its baseline**

```bash
ssh elliot@192.168.86.117 'echo "trainers: $(pgrep -cf "[t]rain.py")"; grep -E "^iter " ~/construct/checkpoints_v9/v9_s20260726.log | tail -1 | cut -c1-140'
```

Expected: exactly 1 trainer. Record steps and sps — this is the baseline for measuring what the second trainer costs.

- [ ] **Step 3: Launch run B**

```bash
ssh elliot@192.168.86.117 'cd ~/construct && mkdir -p checkpoints_v10 && \
  setsid nohup .venv/bin/python scripts/resume_train.py <FORK_CHECKPOINT_FROM_STEP_1> \
    --config configs/train_v9_fromscratch.toml \
    --reward-config configs/reward_v10_jumpboost.toml \
    --checkpoint-dir checkpoints_v10 \
    --entropy-coef 0.006 \
    > checkpoints_v10/v10_s20260729.log 2>&1 & echo "launched"'
```

Substitute the real checkpoint path from Step 1. If `resume_train.py` has no `--reward-config` flag, check `scripts/resume_train.py --help` — the reward tape may be selected via the train config's `reward_config_path`, in which case copy `configs/train_v9_fromscratch.toml` to `configs/train_v10.toml`, point its `reward_config_path` at the v10 tape, and pass that instead.

- [ ] **Step 4: Verify BOTH runs are alive and the new terms appear**

```bash
ssh elliot@192.168.86.117 'echo "trainers (expect 2): $(pgrep -cf "[t]rain.py")"; \
  echo "--- run B first lines ---"; head -25 ~/construct/checkpoints_v10/v10_s20260729.log; \
  echo "--- new terms present? ---"; grep -oE "boost_gained|flip_events|boost_pickup" ~/construct/checkpoints_v10/v10_s20260729.log | sort -u'
```

Expected: 2 trainers, run B logging iterations, and `boost_gained` / `flip_events` appearing. Run B writes to a DIFFERENT checkpoint dir than run A — two trainers on one dir is a known failure mode.

- [ ] **Step 5: Measure the sps cost to run A**

```bash
ssh elliot@192.168.86.117 'grep -E "^iter " ~/construct/checkpoints_v9/v9_s20260726.log | tail -20 | \
  sed -E "s/.*sps ([0-9,]+).*/\1/" | tr -d "," | awk "{s+=\$1;n++} END {print \"run A sps now:\", s/n}"'
```

Compare to the Step 2 baseline. Box has 16 cores at load ~5.3, so expect a modest drop, not a halving. Record the actual figure — do not assume.

- [ ] **Step 6: Commit the launch record**

```bash
cd /home/steamo/Construct-RLBot
git add docs/superpowers/plans/2026-07-29-jump-flip-boost.md
git commit -m "docs(plan): jump/flip/boost v10 fork plan"
```

---

## Verification (after ~200M steps on run B)

Read run B against run A as control. Bench comparisons use CLUSTER means (>=3 checkpoints x 2 seeds), never single checkpoints — between-checkpoint sd is ~0.09 vs ~0.017 across seeds.

| signal | v9 now | fix worked if |
|---|---|---|
| `p_jump` | 2.5e-3 | rises toward 1e-2+ |
| `air_gate_frac` (fixed 400uu instrument) | 0.0002 | clearly nonzero |
| `air_z` >300uu share | 1.67%, flat over 2B | rises, \|t\| > 3 |
| `flip_events` | unmeasured | nonzero at all |
| `boost_gained` | unmeasured | rises vs run A |
| p5 bench cluster | 0.3524 | **must not regress** |

The last row is the guardrail: if jumping appears but the bench drops, run B is abandoned rather than tuned.

Both runs share the box during this period, so any bench is on a loaded box (`gates-need-idle-box` measured load biasing win_share 2-3 points low). Bench run A and run B under the SAME load, or not at all.
