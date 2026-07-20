"""External-bot benchmark: fetch a public RLBot v5 bot, generate a match
config pitting our deploy bot against it, and parse a played match's result
into logs/h2h_history.jsonl using the SAME schema scripts/h2h_eval.py writes.

WHY THIS EXISTS (see docs/external-bench.md, docs/levers-roadmap-2026-07-19.md,
docs/training-journal.md 2026-07-19 ~14:50 -- the KL-anchor kill-switch):
every metric we have -- self-play goals/min, even scripts/h2h_eval.py's
frozen-reference share -- is self-referential: our checkpoints measured
against our own checkpoints. The project spec's P3 exit criterion is
literally "beats Nexto". This script is the first ABSOLUTE ruler: a public,
independently-trained bot (the Necto/Nexto lineage, RLGym community project)
that we did not train and cannot silently regress alongside.

WHAT THIS SCRIPT DOES NOT DO: it does not play matches. RLBot v5 launches
the real Rocket League game (see deploy/README.md: "RLBot only works in
local/offline matches; it launches the game with -rlbot") -- there is no
headless simulation mode, and this repo's dev environment (WSL/Linux) has no
path to a running Rocket League instance. Matches are played by a human
(Elliot) on a Windows box with RLBot v5 + Steam + Rocket League installed,
using the match config this script generates. This script automates
everything AROUND that manual step: fetching+verifying the external bot's
assets, generating the match.toml, and parsing the human-reported (or
script-captured) result into the shared history file. See
docs/external-bench.md for the full plan and exact commands.

WHY NOT scripts/h2h_eval.py --vs-references: that path builds a
construct.league.matches.MatchRunner, which rebuilds OUR OWN candle net
(schema_version/net.heads) from a checkpoint.pt's dict and plays it inside
OUR OWN RocketSim engine (engine/src/*.rs) -- entirely in-process, no real
game. Nexto/Necto ship a *.pt TorchScript module with a completely different
architecture, observation builder and action space, runnable only as an
independent RLBot v5 agent process talking to the real game. There is no way
to load it through MatchRunner without a from-scratch port into candle
(out of scope; see docs/external-bench.md). So this script writes directly
to logs/h2h_history.jsonl via h2h_eval.append_h2h_history (imported, not
duplicated) instead of going through run_vs_references.

REGISTRY / LICENSE: EXTERNAL_BOTS below pins exact upstream files by SHA256
(and the source commit each pin was taken from) so fetches are reproducible
and tamper-evident. The Necto/Nexto weights and code are licensed CC
BY-NC-SA 4.0 by Rolv-Arild/Necto (verified by fetching the LICENSE file
directly -- see docs/external-bench.md) -- NonCommercial, Attribution,
ShareAlike. fetch_external_bot() always writes that LICENSE file alongside
the vendored assets; do not strip it if you move these files elsewhere.

NO NETWORK IN TESTS: only fetch_external_bot() (and its urllib call) touch
the network. Everything else -- checksum verification, match-config
generation, result parsing -- is pure and covered in
tests/python/test_bench_external.py against real upstream byte values or
tmp_path fixtures.
"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = REPO / "deploy" / "external"

# scripts/ isn't a package; import its sibling module the same way
# tests/python/test_h2h.py does. We reuse h2h_eval's history-append + goal
# math instead of re-implementing them, so "same schema" is true by
# construction, not by copy-paste.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import h2h_eval  # noqa: E402

DEFAULT_HISTORY = h2h_eval.DEFAULT_HISTORY

# Rocket League's physics tick rate (see deploy/bot.py: ball_pred_rows()
# samples 120 Hz BallPrediction slices at tau*120; deploy/README's
# maximum_tick_rate_preference convention agrees). Used only to translate a
# real match's wall/game-clock duration into an approximate "steps" figure
# for the history row -- NOT a claim that a live match and an internal
# RocketSim rollout tick at the same granularity in any deeper sense, just a
# unit so the dashboard's "steps/side" tile isn't blank.
TICK_RATE_HZ = 120.0

TEAM_BLUE = 0
TEAM_ORANGE = 1


class AssetChecksumError(ValueError):
    """A fetched (or already-on-disk) external-bot asset doesn't match its
    pinned SHA256 -- refuse to use it rather than silently benchmarking
    against a corrupted download or a tampered/substituted file."""


class MatchResultError(ValueError):
    """A match-result dict is missing required keys or has an invalid
    (non-integer / negative) score -- refuse to parse rather than write a
    garbage row to logs/h2h_history.jsonl."""


# ---------------------------------------------------------------------------
# pure functions: checksums
# ---------------------------------------------------------------------------

def sha256_file(path, chunk_size=1 << 20):
    """SHA256 hex digest of a file on disk, streamed (safe for the ~1.8 MB
    model files without loading them whole)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_asset(path, expected_sha256):
    """Raise AssetChecksumError (mismatch) or FileNotFoundError (missing)
    if `path` doesn't match `expected_sha256`; return the actual digest
    otherwise. The one gate every fetched or pre-existing external-bot file
    passes through before it's considered usable."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"asset missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise AssetChecksumError(
            f"checksum mismatch for {path}: expected {expected_sha256}, got "
            f"{actual} -- refusing to use a corrupted or tampered "
            "external-bot asset (re-run fetch --force to re-download)"
        )
    return actual


# ---------------------------------------------------------------------------
# registry -- pinned upstream files (see docs/external-bench.md for the full
# provenance writeup: repos, license text, dates). Every URL is pinned to a
# specific commit SHA (not a moving branch) so a checksum mismatch always
# means "the download was corrupted/tampered", never "upstream moved on".
# Checksums were computed 2026-07-19 by downloading each file directly and
# running SHA256 over the bytes -- see docs/external-bench.md for the
# verification log.
# ---------------------------------------------------------------------------

_NECTOFAMILY_COMMIT = "0bdb6b49072f6f3829319e68bd6210a0ca4b24a2"  # VirxEC/NectoFamily master, 2025-11-25
_NECTO_COMMIT = "2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca"  # Rolv-Arild/Necto master, 2024-09-17 (LICENSE only)

_NECTOFAMILY_RAW = f"https://raw.githubusercontent.com/VirxEC/NectoFamily/{_NECTOFAMILY_COMMIT}"
_NECTO_RAW = f"https://raw.githubusercontent.com/Rolv-Arild/Necto/{_NECTO_COMMIT}"

# Shared across every bot in EXTERNAL_BOTS: the pip requirements NectoFamily
# ships (torch/numpy/rlgym_compat pinned for RLBot v5) and the upstream
# LICENSE (CC BY-NC-SA 4.0, from the original Rolv-Arild/Necto repo -- the
# VirxEC/NectoFamily v5-port repo carries no LICENSE file of its own, so we
# vendor the upstream one alongside every fetched bot to keep attribution
# and the license terms attached to the weights, per the ShareAlike clause).
SHARED_FILES = {
    "requirements.txt": {
        "url": f"{_NECTOFAMILY_RAW}/requirements.txt",
        "sha256": "d198856fe3f6044a9277f4dae0cb56429978b052fdcccf5d22363d344caac52c",
        "size": 159,
    },
    "LICENSE": {
        "url": f"{_NECTO_RAW}/LICENSE",
        "sha256": "1349a4b6148492b44f629e64eed676612e234fe9a839e4f3b277c1482c8849f1",
        "size": 20849,
    },
}

EXTERNAL_BOTS = {
    "nexto": {
        # Author's own claim (Rolv-Arild/Necto README, verified 2026-07-19):
        # "Approximately Grand Champion 1 level in 1v1, 2v2 and 3v3 (top
        # 0.12%, 0.95%, 0.46% of the playerbase respectively)". NOT
        # independently benchmarked by us -- see docs/external-bench.md.
        "label": "nexto-GC1",
        "description": "Nexto (Necto v2) -- author-claimed ~GC1 in 1v1/2v2/3v3",
        "agent_id": "rlgym/nexto",
        "source_repo": "https://github.com/Rolv-Arild/Necto",
        "port_repo": "https://github.com/VirxEC/NectoFamily",
        "model_file": "nexto-model.pt",
        "files": {
            "bot.toml": {
                "url": f"{_NECTOFAMILY_RAW}/nexto/bot.toml",
                "sha256": "208d8560c245184f53c79a9a1e74f3256b295ba45ebe9598e36d1ecdac360021",
                "size": 929,
            },
            "loadout.toml": {
                "url": f"{_NECTOFAMILY_RAW}/nexto/loadout.toml",
                "sha256": "fe5313e9a75b829fdf8993968fba3b4687cdff21947c309ee451a89ce91f7659",
                "size": 1799,
            },
            "bot.py": {
                "url": f"{_NECTOFAMILY_RAW}/nexto/bot.py",
                "sha256": "e5d7a0a3b49e911b1eedfad23bbf06962899fa15d055024a31819636fbe22565",
                "size": 15589,
            },
            "agent.py": {
                "url": f"{_NECTOFAMILY_RAW}/nexto/agent.py",
                "sha256": "bc37179998ef67d185f12a773481c67501a41cb0ca738fadc08d1e0aeadd69a5",
                "size": 2793,
            },
            "nexto_obs.py": {
                "url": f"{_NECTOFAMILY_RAW}/nexto/nexto_obs.py",
                "sha256": "0a59d8741407cee33d8e198d54806b309e48654bfe275e0e0d02c4224df34ea6",
                "size": 11182,
            },
            "nexto-model.pt": {
                "url": f"{_NECTOFAMILY_RAW}/nexto/nexto-model.pt",
                "sha256": "bf5343b5eeacac6bf7cdb75dac4a5c14ba0f94d820eae75f00a211b6119d69fa",
                "size": 1852625,
            },
        },
    },
    "necto": {
        # Author's own claim: "Around Diamond level" -- a weaker sanity-check
        # tier, not the primary P3 target (Nexto is).
        "label": "necto-diamond",
        "description": "Necto (v1) -- author-claimed ~Diamond",
        "agent_id": "rlgym/necto",
        "source_repo": "https://github.com/Rolv-Arild/Necto",
        "port_repo": "https://github.com/VirxEC/NectoFamily",
        "model_file": "necto-model.pt",
        "files": {
            "bot.toml": {
                "url": f"{_NECTOFAMILY_RAW}/necto/bot.toml",
                "sha256": "2df83edc6bb11aadd3b37490bcbc80a52da3acb859f743b9574964839e3178a9",
                "size": 868,
            },
            "loadout.toml": {
                "url": f"{_NECTOFAMILY_RAW}/necto/loadout.toml",
                "sha256": "39671db96813f4acb3b6a990807b55983576ef65c13e1c8cfea271512ab299b2",
                "size": 1192,
            },
            "bot.py": {
                "url": f"{_NECTOFAMILY_RAW}/necto/bot.py",
                "sha256": "88b8b0743ca0be509b4d9905bb466d24bd971cb550f6f28f1c71ab77f8569a8d",
                "size": 7909,
            },
            "agent.py": {
                "url": f"{_NECTOFAMILY_RAW}/necto/agent.py",
                "sha256": "ebcde2575f2ee64e9342cef683b8304861c82408b1cb2ecd66a8cb430f33bdd7",
                "size": 2299,
            },
            "necto_obs.py": {
                "url": f"{_NECTOFAMILY_RAW}/necto/necto_obs.py",
                "sha256": "ca3bf70988d38ef6be9fa9f12e16959cfbae93a37a6a2f89de6baf75b4461004",
                "size": 4715,
            },
            "necto-model.pt": {
                "url": f"{_NECTOFAMILY_RAW}/necto/necto-model.pt",
                "sha256": "137f94924afb0402af7ae9f12b7f2d7e47adff962afad4dbed7bf864c34c542b",
                "size": 734598,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# pure functions: asset planning (no network -- just describes what
# fetch_external_bot() below would do)
# ---------------------------------------------------------------------------

def asset_plan(bot_name, dest_dir=None):
    """[{url, dest, sha256, size}] for every file a fetch of `bot_name`
    would write -- the bot's own files plus SHARED_FILES (requirements.txt,
    LICENSE), all landing in dest_dir (default EXTERNAL_DIR/bot_name).
    Raises KeyError for an unknown bot. Pure -- no filesystem or network
    access, safe to call from tests."""
    if bot_name not in EXTERNAL_BOTS:
        raise KeyError(
            f"unknown external bot {bot_name!r}; known: {sorted(EXTERNAL_BOTS)}"
        )
    bot = EXTERNAL_BOTS[bot_name]
    dest_dir = Path(dest_dir) if dest_dir is not None else (EXTERNAL_DIR / bot_name)
    plan = []
    for name, spec in {**bot["files"], **SHARED_FILES}.items():
        plan.append({
            "url": spec["url"], "dest": dest_dir / name,
            "sha256": spec["sha256"], "size": spec["size"],
        })
    return plan


# ---------------------------------------------------------------------------
# network: fetch + verify (not unit tested -- see tests/python's module
# docstring; asset_plan/verify_asset above cover the logic this drives)
# ---------------------------------------------------------------------------

def fetch_external_bot(bot_name, dest_dir=None, force=False):
    """Download every file in asset_plan(bot_name, dest_dir), verifying each
    against its pinned checksum before it's considered live. A file already
    on disk with a matching checksum is left alone (skip re-download) unless
    force=True. Downloads land in a temp file first and are only moved into
    place after verify_asset passes -- a checksum failure never leaves a
    half-written file at the destination path. Returns the list of dest
    Paths written or confirmed."""
    plan = asset_plan(bot_name, dest_dir)
    out = []
    for item in plan:
        dest = item["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and not force:
            try:
                verify_asset(dest, item["sha256"])
                out.append(dest)
                continue
            except AssetChecksumError:
                pass  # fall through to re-download
        with urllib.request.urlopen(item["url"]) as resp:
            data = resp.read()
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent))
        try:
            with open(fd, "wb") as f:
                f.write(data)
            verify_asset(tmp_name, item["sha256"])
            Path(tmp_name).replace(dest)
        finally:
            Path(tmp_name).unlink(missing_ok=True)
        out.append(dest)
    return out


# ---------------------------------------------------------------------------
# pure functions: match-config generation
# ---------------------------------------------------------------------------

def _team_index(team):
    """Normalize "blue"/"orange"/0/1 to 0/1 -- accepted interchangeably on
    the CLI and in build_match_config for readability."""
    if isinstance(team, str):
        t = team.strip().lower()
        if t in ("blue", "0"):
            return TEAM_BLUE
        if t in ("orange", "1"):
            return TEAM_ORANGE
        raise ValueError(f"unrecognized team {team!r}; use 'blue'/'orange' or 0/1")
    t = int(team)
    if t not in (TEAM_BLUE, TEAM_ORANGE):
        raise ValueError(f"team must be 0 (blue) or 1 (orange), got {t}")
    return t


def build_match_config(our_bot_toml, external_bot_toml, our_team=TEAM_BLUE,
                        match_length="five_minutes", launcher="steam",
                        game_map_upk="Stadium_P"):
    """A plain dict mirroring deploy/match.toml's structure (verified
    working config: [rlbot], [match], [[cars]], [mutators]) with our deploy
    bot and one external bot as the two [[cars]] -- never a psyonix bot,
    this is a bot-vs-bot skill bench, not deploy/match.toml's solo-practice
    config. `our_team` picks which side our bot plays; the external bot
    always takes the other side. Pure -- both toml paths are just recorded
    as strings, never opened."""
    our_team = _team_index(our_team)
    ext_team = TEAM_ORANGE if our_team == TEAM_BLUE else TEAM_BLUE
    cars = [None, None]
    cars[our_team] = {"config_file": str(our_bot_toml), "team": our_team}
    cars[ext_team] = {"config_file": str(external_bot_toml), "team": ext_team}
    return {
        "rlbot": {"launcher": launcher},
        "match": {"game_mode": "Soccar", "game_map_upk": game_map_upk},
        "cars": cars,
        "mutators": {"match_length": match_length},
    }


def build_both_side_configs(our_bot_toml, external_bot_toml, **kwargs):
    """(cfg_blue, cfg_orange): our bot on team 0 in the first, team 1 in the
    second -- both side orders, same spirit as h2h_eval.play_h2h's "SIDE
    BIAS IS REAL" rule (see that module's docstring). Play both, sum the
    goals via combine_sides() below, don't trust a single match's share."""
    cfg_a = build_match_config(our_bot_toml, external_bot_toml, our_team=TEAM_BLUE, **kwargs)
    cfg_b = build_match_config(our_bot_toml, external_bot_toml, our_team=TEAM_ORANGE, **kwargs)
    return cfg_a, cfg_b


def _toml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)  # double-quoted TOML basic string; our values are plain ASCII paths/enums
    raise TypeError(f"unsupported match-config value type: {type(v)!r} ({v!r})")


def render_match_toml(config):
    """dict -> TOML text. Deliberately minimal (only the shapes
    build_match_config produces: flat [rlbot]/[match]/[mutators] tables plus
    a "cars" list of flat dicts rendered as [[cars]]) -- not a general TOML
    writer. Round-trips through tomllib (see the test file) so the generated
    config is exactly what a hand-written match.toml like deploy/match.toml
    would parse to."""
    lines = []
    for section in ("rlbot", "match"):
        if section not in config:
            continue
        lines.append(f"[{section}]")
        for k, v in config[section].items():
            lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    for car in config.get("cars", []):
        lines.append("[[cars]]")
        for k, v in car.items():
            lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    if "mutators" in config:
        lines.append("[mutators]")
        for k, v in config["mutators"].items():
            lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def write_match_toml(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_match_toml(config))
    return path


# ---------------------------------------------------------------------------
# pure functions: match-result parsing
#
# RESULT JSON FORMAT (ours -- see docs/external-bench.md): RLBot v5's own
# flatbuffer field names for team score could not be verified from this
# (WSL/Linux) dev box -- `rlbot`/`rlgym_compat` are Windows-only deps (same
# limitation deploy/README.md already documents for compat_to_state_dict).
# Rather than guess at unverified flatbuffer internals, this script defines
# its own tiny result schema that Elliot fills in after watching a match
# (final scoreboard has exactly these two numbers):
#   {"blue_score": int, "orange_score": int, "match_length_s": number?}
# match_length_s is optional (defaults the row's "steps" to 0 if omitted).
# ---------------------------------------------------------------------------

def approx_steps_from_duration(seconds):
    """Wall/game-clock seconds -> an approximate physics-tick "steps" count
    (seconds * TICK_RATE_HZ), so the dashboard's "steps/side" tile isn't
    blank for external-bench rows. None or falsy -> 0 (matches h2h_eval's
    own int(e.get("steps") or 0) tolerance in dashboard.py's parser)."""
    if not seconds:
        return 0
    return round(float(seconds) * TICK_RATE_HZ)


def parse_match_result(result, our_team=TEAM_BLUE):
    """result: {"blue_score": int, "orange_score": int, "match_length_s"?}
    -> (goals_ck, goals_ref, steps), where goals_ck is OUR bot's goals and
    goals_ref is the external bot's, per which team our_team says we played.
    Raises MatchResultError on missing/malformed keys -- refuses to guess."""
    our_team = _team_index(our_team)
    try:
        blue = result["blue_score"]
        orange = result["orange_score"]
    except (KeyError, TypeError) as e:
        raise MatchResultError(
            f"match result missing required key(s): {e} -- expected "
            "{'blue_score': int, 'orange_score': int, 'match_length_s'?: number}"
        ) from e
    for name, v in (("blue_score", blue), ("orange_score", orange)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise MatchResultError(f"{name} must be a non-negative int, got {v!r}")
    goals_ck, goals_ref = (blue, orange) if our_team == TEAM_BLUE else (orange, blue)
    steps = approx_steps_from_duration(result.get("match_length_s"))
    return goals_ck, goals_ref, steps


def combine_sides(result_a, result_b, our_team_a=TEAM_BLUE):
    """Two match results, played with sides swapped (our bot on our_team_a
    in `result_a`, on the opposite team in `result_b`) -> h2h_eval's
    aggregate_sides() dict, i.e. exactly what internal h2h's play_h2h
    produces for two side orders. Reuses h2h_eval's own aggregation math
    instead of re-deriving it, so an external-bench two-match result and an
    internal h2h_eval result are combined identically."""
    our_team_a = _team_index(our_team_a)
    goals_ck_a, goals_ref_a, _ = parse_match_result(result_a, our_team_a)
    goals_ck_b, goals_ref_b, _ = parse_match_result(result_b, 1 - our_team_a)
    return h2h_eval.aggregate_sides((goals_ck_a, goals_ref_a), (goals_ck_b, goals_ref_b))


def record_result_row(history_path, ck, bot_name, result, our_team=TEAM_BLUE,
                       result_swapped=None, seed=0, ts=None, ref_label=None):
    """Parse one (or, side-bias-controlled, two swapped-side) match
    result(s) for `ck` vs EXTERNAL_BOTS[bot_name] and append one row to
    history_path via h2h_eval.append_h2h_history -- the exact same function
    (and therefore exact same schema) scripts/h2h_eval.py's
    run_vs_references uses. `ref` is the external bot's vendored model file
    path (so Path(ref).name matches what a human sees in deploy/external/),
    e.g. "deploy/external/nexto/nexto-model.pt".

    Returns (row, warning|None). warning is set when only one match was
    played -- side bias is real (see h2h_eval's module docstring) and a
    single-sided external-bench result hasn't controlled for it."""
    if bot_name not in EXTERNAL_BOTS:
        raise KeyError(f"unknown external bot {bot_name!r}; known: {sorted(EXTERNAL_BOTS)}")
    bot = EXTERNAL_BOTS[bot_name]
    ref = str(EXTERNAL_DIR / bot_name / bot["model_file"])
    label = ref_label if ref_label is not None else bot["label"]

    warning = None
    if result_swapped is not None:
        agg = combine_sides(result, result_swapped, our_team_a=our_team)
        goals_ck, goals_ref, steps = agg["a"], agg["b"], approx_steps_from_duration(
            result.get("match_length_s")
        )
        if agg["unstable"]:
            warning = (
                f"unstable measurement: side orders disagree by "
                f"{agg['disagreement'] * 100:.1f}% -- treat this share with caution "
                "(see h2h_eval.DISAGREEMENT_THRESHOLD)"
            )
    else:
        goals_ck, goals_ref, steps = parse_match_result(result, our_team)
        warning = (
            "single-sided result -- side bias is real (h2h_eval's module "
            "docstring: swings of >40 percentage points have been observed "
            "between side orders on the SAME pair of policies); play the "
            "swapped-side match too and pass --result-swapped before "
            "trusting this share"
        )

    row = h2h_eval.append_h2h_history(
        history_path, ck, ref, label, goals_ck, goals_ref, steps, seed, ts=ts,
    )
    return row, warning


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_list(args):
    for name, bot in EXTERNAL_BOTS.items():
        print(f"{name}: {bot['description']} (agent_id={bot['agent_id']})")
        print(f"    source: {bot['source_repo']}")
        print(f"    v5 port: {bot['port_repo']}")


def _cmd_fetch(args):
    paths = fetch_external_bot(args.bot, dest_dir=args.dest, force=args.force)
    for p in paths:
        print(f"ok: {p}")
    print(f"\n{len(paths)} file(s) verified in {args.dest or (EXTERNAL_DIR / args.bot)}")
    print("LICENSE (CC BY-NC-SA 4.0, from Rolv-Arild/Necto) vendored alongside -- keep it "
          "attached if you move these files.")


def _cmd_gen_match(args):
    our_bot_toml = args.our_bot_toml
    ext_bot_toml = args.ext_bot_toml or str(EXTERNAL_DIR / args.bot / "bot.toml")
    kwargs = dict(match_length=args.match_length)
    cfg_a, cfg_b = build_both_side_configs(our_bot_toml, ext_bot_toml, **kwargs)
    out_dir = Path(args.out_dir)
    path_a = write_match_toml(cfg_a, out_dir / f"match_construct_vs_{args.bot}_blue.toml")
    path_b = write_match_toml(cfg_b, out_dir / f"match_construct_vs_{args.bot}_orange.toml")
    print(f"wrote {path_a} (construct=blue, {args.bot}=orange)")
    print(f"wrote {path_b} (construct=orange, {args.bot}=blue)")
    print("Run BOTH in the RLBot v5 GUI/launcher on Windows (see docs/external-bench.md) "
          "-- side bias is real, don't trust a single match's share.")


def _cmd_record_result(args):
    result = json.loads(Path(args.result).read_text())
    result_swapped = json.loads(Path(args.result_swapped).read_text()) if args.result_swapped else None
    row, warning = record_result_row(
        args.history, args.ck, args.bot, result,
        our_team=args.our_team, result_swapped=result_swapped,
        seed=args.seed, ref_label=args.label,
    )
    print(f"appended to {args.history}: {json.dumps(row)}")
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)


def build_parser():
    ap = argparse.ArgumentParser(
        description="External-bot (Nexto/Necto) benchmark: fetch assets, "
                     "generate an RLBot v5 match config, parse a played "
                     "match's result into logs/h2h_history.jsonl."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list known external bots")
    p_list.set_defaults(func=_cmd_list)

    p_fetch = sub.add_parser("fetch", help="download + checksum-verify an external bot's assets")
    p_fetch.add_argument("bot", choices=sorted(EXTERNAL_BOTS))
    p_fetch.add_argument("--dest", default=None, help="default: deploy/external/<bot>")
    p_fetch.add_argument("--force", action="store_true", help="re-download even if already verified")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_gen = sub.add_parser("gen-match", help="write both-side-order match.toml files")
    p_gen.add_argument("bot", choices=sorted(EXTERNAL_BOTS))
    p_gen.add_argument("--our-bot-toml", default=str(REPO / "deploy" / "bot.toml"))
    p_gen.add_argument("--ext-bot-toml", default=None, help="default: deploy/external/<bot>/bot.toml")
    p_gen.add_argument("--out-dir", default=str(EXTERNAL_DIR))
    p_gen.add_argument("--match-length", default="five_minutes")
    p_gen.set_defaults(func=_cmd_gen_match)

    p_rec = sub.add_parser("record-result", help="parse a played match's result into logs/h2h_history.jsonl")
    p_rec.add_argument("bot", choices=sorted(EXTERNAL_BOTS))
    p_rec.add_argument("--ck", required=True, help="our checkpoint path (labels the row, like h2h_eval)")
    p_rec.add_argument("--result", required=True, help="result JSON: {blue_score, orange_score, match_length_s?}")
    p_rec.add_argument("--result-swapped", default=None, help="optional swapped-side result JSON (recommended)")
    p_rec.add_argument("--our-team", default="blue", help="which team WE played in --result (blue/orange)")
    p_rec.add_argument("--seed", type=int, default=0)
    p_rec.add_argument("--label", default=None, help="default: the bot's registry label, e.g. nexto-GC1")
    p_rec.add_argument("--history", default=str(DEFAULT_HISTORY))
    p_rec.set_defaults(func=_cmd_record_result)

    return ap


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
