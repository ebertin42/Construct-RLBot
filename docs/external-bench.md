# External-bot benchmark (task #54) — an ABSOLUTE skill ruler

## Why this exists

Every metric this project has produced so far is self-referential: self-play
goals/min (`scripts/eval_metrics.py`) measures our policy against a mirror of
*itself*, and even `scripts/h2h_eval.py`'s frozen-reference goal share
measures our current checkpoint against our own *past* checkpoints
(`configs/h2h_references.toml`). That confound is not hypothetical — it
already misled us once: `docs/training-journal.md` (2026-07-19 ~14:50, "the
KL-anchor kill-switch") records self-play goals/min hiding an 800M-step
regression and making a 3.5x-weaker policy *look* like it was improving,
which is the entire reason `h2h_eval.py` exists. But h2h_eval's own frozen
references are still just our best checkpoint so far — a relative ruler that
can only ever measure "better than us, yesterday."

The project spec's P3 exit criterion is literally "beats Nexto." We have had
no way to measure distance to that criterion. This task builds one: fetch a
public, independently-trained bot we did not train and cannot silently
regress alongside, generate a match config pitting our deploy bot against
it, and parse the result into the same history file and schema the
dashboard already plots.

## What was verified vs. inferred

Everything below marked **verified** was checked against a primary source
(a GitHub repo's actual file content, commit metadata via the GitHub API, or
a wiki page's own quoted text) during this task, 2026-07-19. Nothing here is
taken from a single blog post.

### The bot: Necto / Nexto (RLGym community project)

- **Source repo**: [Rolv-Arild/Necto](https://github.com/Rolv-Arild/Necto)
  — **verified**: not archived, 301 stars, last pushed 2024-09-17 (the
  project is dormant but the repo is live and public). Its own README
  states, verbatim:

  > V1: Necto - Around Diamond level.
  > V2: Nexto - Approximately Grand Champion 1 level in 1v1, 2v2 and 3v3
  > (top 0.12%, 0.95%, 0.46% of the playerbase respectively)
  > V3: Tecko - Canceled due to lack of improvement.
  > The project is not being worked on anymore.

  These are the **author's own claims**, not a rank we independently
  verified — treat "Nexto ≈ GC1" as a documented prior, not a measured
  fact, until our own head-to-head results say otherwise. **Tecko was never
  released** (canceled) — do not go looking for it. The README also
  mentions "Nexto+", a private post-training upgrade that "is not available
  for play." Only Necto and Nexto (and a same-skill "Toxic Nexto" variant)
  are actually obtainable.
- **License — verified by fetching the LICENSE file directly**: **CC
  BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike
  4.0 International). Non-commercial use with attribution and share-alike
  is permitted; this project's internal benchmarking is non-commercial
  research, so fetching and running these weights is compliant. GitHub's
  own license detector reports `"other"/NOASSERTION` for this repo (and for
  the RLBot v5 botpack below) — expected, since GitHub's detector is tuned
  for OSI software licenses, not Creative Commons content licenses; it is
  not a sign the CC BY-NC-SA text I fetched is wrong.
- **The v4 rlbot-support/ folder in this repo uses the legacy `bot.cfg`
  format and does NOT run under the RLBot v5 launcher** — see below.

### The RLBot v5 port: VirxEC/NectoFamily

- **Verified via `.gitmodules`**: the *official* RLBot v5 community pack,
  [RLBot/botpack](https://github.com/RLBot/botpack) (not archived, pushed
  2026-06-15), includes Necto/Nexto as a git submodule pointing directly at
  [VirxEC/NectoFamily](https://github.com/VirxEC/NectoFamily). This is the
  community-blessed v5 port, even though the port repo itself is small (not
  archived, pushed 2025-11-25, 1 star — expected for a submodule dependency,
  not a standalone project).
- **Caveat — verified**: NectoFamily's GitHub API metadata shows
  `"license": null`. The port carries no license file of its own for the
  glue code (bot.toml, the RLBot v5 `Bot` wrapper, the PyInstaller build
  scaffolding). The underlying weights and original obs-builder/agent logic
  remain governed by Necto's CC BY-NC-SA 4.0 (attribution fields — developer,
  source_link — are preserved as-is in the vendored `bot.toml`, satisfying
  the Attribution clause). `scripts/bench_external.py` always vendors the
  original upstream `LICENSE` file alongside every fetched asset; **do not
  strip it if you move these files anywhere else.**
- **Verified — this is a genuine RLBot v5 bot folder**, structurally
  identical in shape to our own `deploy/`: `nexto/bot.toml` uses the same
  `#:schema https://rlbot.org/schemas/agent.json` schema as `deploy/bot.toml`,
  with `agent_id = "rlgym/nexto"` and `run_command = "uv run bot.py"`. The
  weights (`nexto-model.pt`, 1,852,625 bytes; `necto-model.pt`, 734,598
  bytes) are `torch.jit.load`-able TorchScript archives committed directly
  in the repo (not git-lfs pointers — confirmed by matching GitHub API
  `size` to the actual byte count).
- **Verified — the RLBot v5 wiki itself uses Necto as its own worked
  example**: [wiki.rlbot.org/v5/botmaking/config-files/](https://wiki.rlbot.org/v5/botmaking/config-files/)'s
  full example `match.toml` literally includes `config_file =
  "necto/bot.toml"` as a `[[cars]]` entry. This is the community-canonical
  way to put Nexto/Necto into a v5 match — not a novel integration we're
  inventing.
- **Dependency note**: NectoFamily's `requirements.txt` pins `rlbot>=2.0.0.beta`,
  `numpy==2.*`, `rlgym_compat @ git+https://github.com/JPK314/rlgym-compat`,
  `torch==2.4.1+cpu` — the **same** `rlgym_compat` fork our own
  `deploy/requirements.txt` uses, but an unpinned ref vs. our pinned commit
  `50b70c87b9af7bbce46f4d939d85e9117d2dc3c0`, and a pinned `torch==2.4.1+cpu`
  vs. our own unpinned `torch`. **Use a separate venv for the external bot folder** — RLBot v5's
  per-bot `run_command`/`run_command_linux` supports this natively (each
  `[[cars]]` entry's `config_file` points at its own bot's own interpreter),
  so there is no need to reconcile the two dependency sets.

### RLBot v5 itself

- **Verified**: "first-class Windows & Linux support"; macOS is explicitly
  unsupported ("there's no way to run Rocket League without a full VM").
  RLBot v5 **requires the actual Rocket League game** — there is no headless
  simulation mode. This matches `deploy/README.md`'s existing statement:
  "RLBot only works in local/offline matches; it launches the game with
  -rlbot."
- **Implication for this repo's dev environment**: this box is WSL/Linux
  without a Steam+Rocket League install or a path to launch/render the game.
  **No external-bot match can be played from here.** RLBot v5's Linux
  support assumes a real Linux desktop with a GPU and Steam — not WSL.
  Matches must be played on Elliot's Windows box, using the existing
  `deploy/README.md` setup (RLBot v5 launcher + Python 3.11 + a venv), the
  same way our own bot is already deployed there.
- **Not independently verified from this box** (same limitation
  `deploy/README.md` already documents for `compat_to_state_dict`):
  the exact flatbuffer field names for reading team scores / match end
  state (`packet.teams[i].score`-style access) — `rlbot`/`rlgym_compat` are
  Windows-only dependencies that cannot be installed here to check. The wiki
  confirms the GamePacket carries "current scores and time left" and that
  `packet.match_info.match_phase` compares against `MatchPhase.Ended`, but a
  verbatim field-by-field example was not available. Because of this,
  `scripts/bench_external.py` does **not** attempt to parse RLBot's own
  packet/telemetry format — see "Result flow" below.

## What's automated vs. what's manual

| Step | Automated? | How |
|---|---|---|
| Fetch + checksum-verify Nexto/Necto assets | **Yes** | `scripts/bench_external.py fetch nexto` |
| Generate a match config (both side orders) | **Yes** | `scripts/bench_external.py gen-match nexto` |
| Install the external bot's own Python deps | Manual | `pip install -r deploy/external/nexto/requirements.txt` in a **separate** venv, on Windows |
| Launch RLBot v5 and play the match | **Manual — requires Elliot's Windows box** | RLBot v5 GUI, load the generated match.toml, start match |
| Read the final scoreboard | Manual | Elliot watches/replays the match, notes blue/orange score |
| Record the result | **Yes** | `scripts/bench_external.py record-result nexto --ck ... --result result.json --result-swapped result2.json` |
| Plot it on the dashboard | **Yes — zero changes needed** | existing `scripts/dashboard.py` "Head-to-head (skill)" panel reads `logs/h2h_history.jsonl` unmodified |

The genuinely automatable third of this task (asset fetch/verify, match-config
generation, result parsing) is fully built and tested
(`tests/python/test_bench_external.py`, all pure functions, no network). The
match itself cannot be automated or run from this environment — RLBot v5
launches the real game, and there is no headless path. **This is not a
missing feature to build around; it is a hard constraint of the RLBot v5
architecture**, verified above.

### Why `h2h_eval.py --vs-references` can't just point at Nexto

`h2h_eval.py`'s `run_vs_references` builds a
`construct.league.matches.MatchRunner`, which reads `{schema_version,
config.net.heads, total_steps}` out of a checkpoint's own `torch.save`d
dict, rebuilds *our* candle net from it, and plays it inside *our* RocketSim
engine (`engine/src/*.rs`) — entirely in-process, no real game involved.
Nexto's `nexto-model.pt` is a `torch.jit.load` TorchScript module with a
totally different architecture, observation builder (`nexto_obs.py`, ~11 KB)
and action space (a hand-built 90-row discrete lookup table — coincidentally
the same row count as our v0 action table, but a different table: different
throttle/steer/aerial groupings, verified by reading `nexto/agent.py`'s
`make_lookup_table()`). `checkpoint_meta()` would not even load it cleanly,
let alone play it through `MatchRunner`. Porting Nexto's architecture into
candle so it could
run inside our internal engine is a real option in principle but a
significant, separate undertaking (reverse-engineering its obs/action
contracts and net shape) that was explicitly out of scope for this task —
see `configs/h2h_references.toml`'s comment block for why the reference slot
stays commented out rather than pointing at a fake bridge.

### Result flow

1. Elliot runs `scripts/bench_external.py fetch nexto` (network; downloads +
   checksum-verifies into `deploy/external/nexto/`, vendors the LICENSE).
2. Elliot runs `scripts/bench_external.py gen-match nexto` to write
   `deploy/external/match_construct_vs_nexto_blue.toml` and
   `..._orange.toml` — both side orders, mirroring `h2h_eval.py`'s own
   "SIDE BIAS IS REAL" rule (its module docstring documents swings of >40
   percentage points on the *same* pair of policies from side order alone).
3. On the Windows box: `pip install -r deploy/external/nexto/requirements.txt`
   into a separate venv for the external bot folder (do not reuse
   `deploy/`'s venv — see the dependency note above); load each match.toml
   in the RLBot v5 GUI and play it.
4. Elliot writes down the final scoreboard as a small JSON file per match —
   `{"blue_score": int, "orange_score": int, "match_length_s": number}` —
   this repo's own result schema (see `scripts/bench_external.py`'s
   `parse_match_result` docstring for why it isn't RLBot's own flatbuffer
   format: that format could not be verified from this box).
5. `scripts/bench_external.py record-result nexto --ck
   checkpoints_entity/ck_XXXXXXXXXX.pt --result blue.json --result-swapped
   orange.json` combines both sides (reusing `h2h_eval.aggregate_sides` —
   literally imported, not reimplemented) and appends ONE row to
   `logs/h2h_history.jsonl` via `h2h_eval.append_h2h_history` — the same
   function, same schema (`{ts, ck, ref, ref_label, goals_ck, goals_ref,
   share, steps, seed}`) that `h2h_eval.py --vs-references` writes.
6. `scripts/dashboard.py`'s existing "Head-to-head (skill)" panel picks up
   the new `ref_label` ("nexto-GC1" / "necto-diamond") automatically — no
   dashboard changes needed, confirmed by
   `test_record_result_row_appended_lines_are_valid_jsonl_and_dashboard_parseable`
   round-tripping a generated row through `dashboard.parse_h2h_history`.

`steps` for an external match is an *approximation* — game-clock seconds ×
120 Hz (Rocket League's physics tick rate, the same constant
`deploy/bot.py`'s `ball_pred_rows` uses for its 120 Hz `BallPrediction`
slices) — not a claim that a real match and an internal RocketSim rollout
tick at the same granularity in any deeper sense. It only exists so the
dashboard's "steps/side" tile isn't blank.

## Exact commands

```bash
# 1. Fetch + verify (run on whichever box has network; the resulting files
#    can be copied to the Windows box like deploy/ already is)
.venv/bin/python scripts/bench_external.py fetch nexto
.venv/bin/python scripts/bench_external.py fetch necto   # optional, weaker sanity tier

# 2. Generate both-side-order match configs
.venv/bin/python scripts/bench_external.py gen-match nexto

# --- on Elliot's Windows box, in the RLBot v5 GUI ---
#   pip install -r deploy\external\nexto\requirements.txt   (separate venv)
#   Load deploy\external\match_construct_vs_nexto_blue.toml, start match, note score
#   Load deploy\external\match_construct_vs_nexto_orange.toml, start match, note score

# 3. Record the result (either box, no network needed)
python scripts/bench_external.py record-result nexto \
    --ck checkpoints_entity/ck_000909000000.pt \
    --result blue_result.json --result-swapped orange_result.json \
    --seed 11
```

Where `blue_result.json` looks like:

```json
{"blue_score": 4, "orange_score": 2, "match_length_s": 300}
```

## Tiers

- **`nexto`** (primary): author-claimed ~GC1 in 1v1/2v2/3v3. This is the
  bot the P3 spec means by "beats Nexto."
- **`necto`** (secondary, weaker sanity check): author-claimed ~Diamond. A
  policy that can't beat Necto shouldn't be expected to beat Nexto — useful
  as an earlier, cheaper checkpoint on the ladder.
- **Not built**: `nexto_toxic` (same skill as Nexto, different personality —
  no skill-tier value over `nexto`, so left out of `EXTERNAL_BOTS` to keep
  the registry small; trivial to add later by copying the `nexto` entry's
  shape). An automated RLBot v5 "observer script" that reads the live score
  and writes the result JSON automatically (instead of Elliot typing two
  numbers) is a plausible follow-up but was **not built** — it would need
  live verification of RLBot v5's flatbuffer score fields on Windows, which
  this task could not do from WSL/Linux; see the "Not independently
  verified" note above. Don't assume it exists.

## Files

- `scripts/bench_external.py` — fetch/verify, match-config generation,
  result parsing (pure functions, tested, no network in tests).
- `tests/python/test_bench_external.py` — 54 tests, all offline.
- `deploy/external/` — fetch destination, gitignored (large binary weights
  + third-party code; not committed, reproducible from the pinned registry
  in `scripts/bench_external.py`).
- `configs/h2h_references.toml` — a comment block explaining why no live
  external-bot `[[reference]]` entry was added (see "Why `h2h_eval.py
  --vs-references` can't just point at Nexto" above); the slot stays
  documented but commented out.
