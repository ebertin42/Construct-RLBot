"""Champion gating: a checkpoint must BEAT the incumbent to become the reference.

WHY THIS EXISTS (docs/training-journal.md, 2026-07-19 ~17:40 -> 2026-07-20
~00:30). Five controlled 20M-step arms were run from ck_000320471040,
identical but for one variable each, and measured head-to-head against their
own starting checkpoint:

    arm A  ent .01,  no league, v4.1          22.0-25.5%
    arm B  ent .001, no league, v4.1          10.4-11.4%
    arm C  ent .003, league 0.5, clean pool   27.1%
    arm D  ent .01,  no league, v3 + ANCHOR   49.2%
    null control (320M vs itself)             51.1%, rerun variance ~3%

Four of the five arms lost 73-89% of their goals to the checkpoint they
started from. PPO self-play in this project has never been demonstrated to
improve the policy, so "newest checkpoint" and "highest ep_rew" are both
worthless as a promotion signal -- ep_rew ROSE in the arms whose skill
collapsed (journal ~21:50). The only trustworthy signal is a measured match
against the current best. This script makes that measurement the gate.

THE CHAMPION is the single reference used as (a) the KL / self-distillation
anchor -- arm D says never train unanchored -- (b) the league opponent-pool
seed, and (c) the deploy candidate. It lives in configs/champion.toml.

BOTH SIDE ORDERS ARE MANDATORY. A single mr.play() is not a measurement: the
null control swung ~+/-6 goals from parity between orders, ~10 goals of spread
(journal ~22:00). Everything here goes through h2h_eval.play_h2h, which plays
and sums both orders; the match/aggregation/schema-guard logic is IMPORTED
from scripts/h2h_eval.py, never reimplemented, so the gate and the ad hoc
harness can never drift apart.

HISTORY rows (logs/champion_history.jsonl) are a superset of h2h_eval's jsonl
schema (ts/ck/ref/ref_label/goals_ck/goals_ref/share/steps/seed), so
dashboard.parse_h2h_history can read this file unchanged and plot gate results
alongside the existing h2h series.

CLI:
    champion_gate.py status
    champion_gate.py gate CK [--promote-if-pass]
    champion_gate.py watch [--auto-promote]
    champion_gate.py promote CK --reason "..."
    champion_gate.py reject  CK --reason "..."

Manual promote/reject are Elliot's hand-pick escape hatch: they record
manual=true plus the reason, so the audit trail always distinguishes a
measured promotion from a judgment call.
"""
import argparse
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# scripts/ isn't a package; import its sibling the same way tests/ and ctl.py
# do. IMPORT, don't copy -- the both-orders logic must have exactly one
# implementation in this repo.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2h_eval  # noqa: E402
from h2h_eval import (  # noqa: E402
    SchemaMismatchError, aggregate_sides, checkpoint_meta, format_result,
    goal_share, play_h2h, require_compatible,
)

DEFAULT_CONFIG = REPO / "configs" / "champion.toml"

PASS = "PASS"
FAIL = "FAIL"


class ChampionConfigError(RuntimeError):
    """configs/champion.toml is missing, unparseable, or missing a required
    key. ALWAYS fatal, never defaulted around: silently inventing a champion
    pointer would mean anchoring training and seeding the league from an
    unmeasured checkpoint, which is precisely the failure mode this whole
    system exists to prevent."""


# ---------------------------------------------------------------------------
# config (pure)
# ---------------------------------------------------------------------------

_REQUIRED = ("champion_ck", "promote_threshold")
_REQUIRED_MIN_GAMES = ("steps", "arenas", "seed", "min_total_goals")
_REQUIRED_WATCH = ("candidate_dir", "poll_seconds", "consecutive_rejects_before_alert")
_REQUIRED_LEAGUE = ("registry", "run", "reward_config")


def load_config(path=DEFAULT_CONFIG):
    """Parse configs/champion.toml. Raises ChampionConfigError -- loudly, with
    the offending path -- for a missing file, malformed toml, a missing
    required key, or a threshold outside (0, 1]. Never returns a default."""
    path = Path(path)
    if not path.is_file():
        raise ChampionConfigError(
            f"champion config not found: {path} -- refusing to guess a champion. "
            f"Create it (see configs/champion.toml in git) before gating."
        )
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ChampionConfigError(f"champion config {path} is corrupt: {e}") from e

    for key in _REQUIRED:
        if key not in raw:
            raise ChampionConfigError(f"champion config {path} is missing required key '{key}'")
    for section, keys in (("min_games", _REQUIRED_MIN_GAMES),
                          ("watch", _REQUIRED_WATCH),
                          ("league", _REQUIRED_LEAGUE)):
        if section not in raw or not isinstance(raw[section], dict):
            raise ChampionConfigError(
                f"champion config {path} is missing required section [{section}]")
        for key in keys:
            if key not in raw[section]:
                raise ChampionConfigError(
                    f"champion config {path} is missing required key '{section}.{key}'")

    thr = float(raw["promote_threshold"])
    if not 0.0 < thr <= 1.0:
        raise ChampionConfigError(
            f"champion config {path}: promote_threshold={thr} is not a goal share in (0, 1]")
    if thr <= 0.5:
        raise ChampionConfigError(
            f"champion config {path}: promote_threshold={thr} is at or below parity. "
            f"The null control (identical policies) measured 51.1%, so a threshold "
            f"<= 0.5 would promote noise -- see the comment in configs/champion.toml."
        )

    cfg = {
        "path": path,
        "champion_ck": str(raw["champion_ck"]),
        "promote_threshold": thr,
        "steps": int(raw["min_games"]["steps"]),
        "arenas": int(raw["min_games"]["arenas"]),
        "seed": int(raw["min_games"]["seed"]),
        "min_total_goals": int(raw["min_games"]["min_total_goals"]),
        "candidate_dir": str(raw["watch"]["candidate_dir"]),
        "poll_seconds": int(raw["watch"]["poll_seconds"]),
        "consecutive_rejects_before_alert": int(raw["watch"]["consecutive_rejects_before_alert"]),
        "registry": str(raw["league"]["registry"]),
        "league_run": str(raw["league"]["run"]),
        "reward_config": str(raw["league"]["reward_config"]),
        "history": str(raw.get("history", {}).get("path", "logs/champion_history.jsonl")),
    }
    return cfg


# --- atomic champion pointer update -----------------------------------------

_CHAMPION_LINE = re.compile(r'^([ \t]*champion_ck[ \t]*=[ \t]*).*$', re.MULTILINE)


def _rewrite_champion_text(text, new_ck):
    """Replace the champion_ck value in `text`, preserving every comment and
    every other key. Raises if the key isn't found (a config we can't rewrite
    is a config we refuse to half-write)."""
    replacement, n = _CHAMPION_LINE.subn(
        lambda m: f'{m.group(1)}{json.dumps(str(new_ck))}', text, count=1)
    if n != 1:
        raise ChampionConfigError(
            "could not locate a 'champion_ck = ...' line to rewrite -- refusing to "
            "rewrite the champion config blind")
    return replacement


def stage_champion_ck(config_path, new_ck):
    """Step 1 of the atomic update: write the new config to a sibling .tmp and
    return its path. The live config is UNTOUCHED at this point -- a crash
    here leaves the old champion perfectly intact, which is the whole reason
    the write is split in two."""
    config_path = Path(config_path)
    text = config_path.read_text()
    new_text = _rewrite_champion_text(text, new_ck)
    # Parse before committing: a rewrite that produced invalid toml must never
    # reach the real path.
    parsed = tomllib.loads(new_text)
    if str(parsed.get("champion_ck")) != str(new_ck):
        raise ChampionConfigError("staged champion config did not round-trip; aborting")
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp.write_text(new_text)
    return tmp


def commit_staged(tmp_path, config_path):
    """Step 2: os.replace, which is atomic on the same filesystem. Either the
    old file or the new file is visible to a concurrent reader; never a
    half-written one."""
    os.replace(tmp_path, config_path)
    return Path(config_path)


def write_champion_ck(config_path, new_ck):
    """stage + commit. Split so tests can simulate a crash between them."""
    return commit_staged(stage_champion_ck(config_path, new_ck), config_path)


# ---------------------------------------------------------------------------
# verdict logic (pure)
# ---------------------------------------------------------------------------

def evaluate(share, total_goals, threshold, min_total_goals):
    """Decide PASS/FAIL for a measured gate match, with the reason.

    Rules, in order:
      * no goals at all -> FAIL (a 0-0 match has no share; it measured nothing)
      * fewer than min_total_goals combined -> FAIL (sample too small to move
        the pointer that anchors training)
      * an exact tie -> FAIL. Explicit even though 0.5 < threshold always
        (load_config rejects thresholds <= 0.5): "the incumbent keeps the belt
        on a draw" is a rule, not an accident of arithmetic.
      * share >= threshold -> PASS, else FAIL.

    Returns (verdict, reason).
    """
    if share is None:
        return FAIL, "no goals scored by either side (0-0) -- nothing measured"
    if total_goals < min_total_goals:
        return FAIL, (f"only {total_goals} combined goals < min_total_goals="
                      f"{min_total_goals} -- sample too small to promote on")
    if share == 0.5:
        return FAIL, "exact tie (50.0%) -- the incumbent keeps the belt on a draw"
    if share >= threshold:
        return PASS, f"share {share * 100:.1f}% >= threshold {threshold * 100:.1f}%"
    return FAIL, f"share {share * 100:.1f}% < threshold {threshold * 100:.1f}%"


def should_promote(verdict, promote_if_pass):
    """Promotion needs BOTH a PASS and explicit opt-in. `gate` without
    --promote-if-pass is a measurement, never a mutation."""
    return bool(verdict == PASS and promote_if_pass)


def consecutive_rejects(rows):
    """How many of the most recent rows in a row are FAILs. Manual rows count:
    a human rejecting three candidates running is exactly as much of a signal
    as the gate doing it."""
    n = 0
    for row in reversed(rows):
        if row.get("verdict") == FAIL:
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------------------
# history jsonl (pure)
# ---------------------------------------------------------------------------

def history_row(candidate, champion, goals_c, goals_champ, verdict, promoted,
                reason, manual=False, steps=0, seed=0, arenas=0, threshold=None,
                share=None, ts=None):
    """One gate decision. The first block of keys is h2h_eval's schema verbatim
    (ck/ref/ref_label/goals_ck/goals_ref/share/steps/seed) so
    dashboard.parse_h2h_history reads this file as-is; the rest is the gate's
    own audit trail."""
    if share is None:
        share = goal_share(goals_c or 0, goals_champ or 0)
    return {
        "ts": int(ts if ts is not None else time.time()),
        # --- h2h_eval-compatible view -------------------------------------
        "ck": Path(candidate).name,
        "ref": Path(champion).name,
        "ref_label": "champion",
        "goals_ck": int(goals_c or 0),
        "goals_ref": int(goals_champ or 0),
        "share": share,
        "steps": int(steps),
        "seed": int(seed),
        # --- gate audit trail ---------------------------------------------
        "candidate": str(candidate),
        "champion": str(champion),
        "goals_c": int(goals_c or 0),
        "goals_champ": int(goals_champ or 0),
        "verdict": verdict,
        "promoted": bool(promoted),
        "reason": reason,
        "manual": bool(manual),
        "arenas": int(arenas),
        "threshold": threshold,
    }


def append_history(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read_history(path):
    """Rows in file order. Unparseable lines are skipped rather than fatal --
    a truncated tail (crash mid-append) must not make `status` unusable."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "ts" in row:
            rows.append(row)
    return rows


def promotions(rows):
    return [r for r in rows if r.get("promoted")]


def gated_candidates(rows):
    """Set of candidate paths that already have a verdict recorded."""
    return {str(r.get("candidate")) for r in rows if r.get("candidate")}


def pending_candidates(candidate_dir, rows, champion_ck, pattern="ck_*.pt"):
    """Checkpoints in candidate_dir that have never been gated, oldest mtime
    first. The current champion is never its own candidate."""
    d = Path(candidate_dir)
    if not d.is_dir():
        return []
    seen = gated_candidates(rows)
    champ = Path(champion_ck).resolve() if champion_ck else None
    out = []
    for p in d.glob(pattern):
        if str(p) in seen:
            continue
        if champ is not None and p.resolve() == champ:
            continue
        out.append(p)
    out.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return out


# ---------------------------------------------------------------------------
# engine-touching glue (mocked out in the tests via run_match)
# ---------------------------------------------------------------------------

def run_match(candidate, champion, steps, arenas, seed):
    """Play candidate vs champion, BOTH SIDE ORDERS, and return h2h_eval's
    aggregate. Raises SchemaMismatchError before building an engine if the two
    checkpoints can't legally play. Tests monkeypatch this whole function; the
    both-orders logic itself lives in h2h_eval.play_h2h."""
    meta_c = checkpoint_meta(candidate)
    meta_champ = checkpoint_meta(champion)
    require_compatible(meta_c, meta_champ)
    mr = h2h_eval._build_runner(meta_c, arenas, seed)
    result = play_h2h(mr, h2h_eval.load_sd(candidate), h2h_eval.load_sd(champion), steps)
    result["meta_candidate"] = meta_c
    result["meta_champion"] = meta_champ
    return result


def add_to_league(cfg, ck, steps, schema_version):
    """Append the new champion to the league pool so the opponent pool tracks
    the champion lineage. Registry.add is a no-op for a ck already present."""
    from construct.league.registry import Registry
    reg = Registry(path=cfg["registry"])
    reg.add(ck, int(steps), cfg["league_run"], cfg["reward_config"],
            schema_version=int(schema_version))
    return reg


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def _print_alert(cfg, rows):
    n = consecutive_rejects(rows)
    limit = cfg["consecutive_rejects_before_alert"]
    if n >= limit:
        print(f"\n!! ALERT: {n} consecutive rejects (limit {limit}). Nothing has "
              f"beaten the champion in {n} attempts -- the run producing these "
              f"candidates is very likely degrading (journal 2026-07-19 ~17:40).",
              file=sys.stderr)
        return True
    return False


def cmd_status(cfg):
    rows = read_history(cfg["history"])
    champ = cfg["champion_ck"]
    champ_path = Path(champ)
    exists = "" if champ_path.is_file() else "   (MISSING ON DISK)"
    print(f"=== champion ===")
    print(f"  {champ}{exists}")
    print(f"  threshold {cfg['promote_threshold'] * 100:.1f}%  "
          f"match {cfg['steps']} steps/side x {cfg['arenas']} arenas, seed {cfg['seed']}, "
          f"both side orders")
    print(f"  config {cfg['path']}")

    proms = promotions(rows)
    print(f"\n=== promotion history ({len(proms)}) ===")
    if not proms:
        print("  (none -- champion is the seeded reference, never promoted through the gate)")
    for r in proms:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        share = r.get("share")
        share_s = f"{share * 100:.1f}%" if isinstance(share, (int, float)) else "n/a"
        kind = "MANUAL" if r.get("manual") else "gated"
        print(f"  {stamp}  {kind:6s}  {Path(str(r.get('candidate'))).name}  "
              f"share={share_s}  {r.get('reason', '')}")

    print(f"\n=== recent gate results ({len(rows)} total) ===")
    for r in rows[-10:]:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        share = r.get("share")
        share_s = f"{share * 100:5.1f}%" if isinstance(share, (int, float)) else "  n/a"
        flags = []
        if r.get("manual"):
            flags.append("manual")
        if r.get("promoted"):
            flags.append("PROMOTED")
        flag_s = (" [" + ",".join(flags) + "]") if flags else ""
        print(f"  {stamp}  {r.get('verdict'):4s}  {share_s}  "
              f"{Path(str(r.get('candidate'))).name}{flag_s}")
    if not rows:
        print("  (none)")

    pend = pending_candidates(cfg["candidate_dir"], rows, cfg["champion_ck"])
    print(f"\n=== pending candidates in {cfg['candidate_dir']} ({len(pend)}) ===")
    for p in pend[-10:]:
        print(f"  {p}")
    if len(pend) > 10:
        print(f"  ... and {len(pend) - 10} older")
    if not pend:
        print("  (none)")

    _print_alert(cfg, rows)
    return 0


def gate_one(cfg, candidate, promote_if_pass, quiet=False):
    """Measure `candidate` against the champion and record the verdict.
    Returns the history row. Promotes only on PASS *and* promote_if_pass."""
    champion = cfg["champion_ck"]
    if not quiet:
        print(f"=== GATE: {Path(candidate).name} vs champion {Path(champion).name} ===")
        print(f"    {cfg['steps']} steps/side x {cfg['arenas']} arenas, seed {cfg['seed']}, "
              f"both side orders (mandatory -- single-order swings ~+/-6 goals)")

    result = run_match(candidate, champion, cfg["steps"], cfg["arenas"], cfg["seed"])
    goals_c, goals_champ = result["a"], result["b"]
    share = result["share"]
    total = goals_c + goals_champ
    verdict, reason = evaluate(share, total, cfg["promote_threshold"], cfg["min_total_goals"])

    if not quiet:
        print(format_result(Path(candidate).name, "champion", result))
        print(f"verdict: {verdict}  ({reason})")

    promoted = should_promote(verdict, promote_if_pass)
    if verdict == PASS and not promote_if_pass:
        reason += " -- not promoted: --promote-if-pass not given"
        if not quiet:
            print("  PASS but NOT promoted: rerun with --promote-if-pass to move the pointer.")

    if promoted:
        meta = result.get("meta_candidate") or checkpoint_meta(candidate)
        _do_promote(cfg, candidate, meta)
        if not quiet:
            print(f"  PROMOTED: champion_ck -> {candidate}")

    row = history_row(
        candidate, champion, goals_c, goals_champ, verdict, promoted, reason,
        manual=False, steps=cfg["steps"], seed=cfg["seed"], arenas=cfg["arenas"],
        threshold=cfg["promote_threshold"], share=share,
    )
    append_history(cfg["history"], row)
    if not quiet:
        print(f"  recorded -> {cfg['history']}")
    return row


def _do_promote(cfg, candidate, meta):
    """Move the pointer atomically, then register the new champion in the
    league pool. Pointer first: if the league append fails, the champion is
    still consistent and the append can be retried; the reverse order could
    leave the pool ahead of the pointer."""
    write_champion_ck(cfg["path"], candidate)
    try:
        add_to_league(cfg, str(candidate), meta.get("steps", 0), meta.get("schema_version", 1))
    except Exception as e:  # noqa: BLE001 -- never lose a valid promotion to a league write
        print(f"  warning: champion promoted but league registry append failed: {e}",
              file=sys.stderr)


def cmd_gate(cfg, candidate, promote_if_pass):
    try:
        gate_one(cfg, candidate, promote_if_pass)
    except SchemaMismatchError as e:
        print(f"gate refused: {e}", file=sys.stderr)
        return 1
    rows = read_history(cfg["history"])
    _print_alert(cfg, rows)
    return 0


def cmd_manual(cfg, candidate, reason, promote):
    """promote/reject: Elliot's hand-pick, regardless of measured share."""
    if not reason:
        print("--reason is required for a manual override (it IS the audit trail)",
              file=sys.stderr)
        return 1
    champion = cfg["champion_ck"]
    if promote:
        meta = checkpoint_meta(candidate)
        _do_promote(cfg, candidate, meta)
        print(f"MANUAL PROMOTE: champion_ck -> {candidate}")
    else:
        print(f"MANUAL REJECT: {candidate} (champion stays {champion})")
    row = history_row(
        candidate, champion, 0, 0, PASS if promote else FAIL, promote, reason,
        manual=True, steps=0, seed=0, arenas=0,
        threshold=cfg["promote_threshold"], share=None,
    )
    append_history(cfg["history"], row)
    print(f"  reason: {reason}")
    print(f"  recorded (manual=true) -> {cfg['history']}")
    return 0


def cmd_watch(cfg, auto_promote, once=False, sleep=time.sleep):
    """Poll candidate_dir and gate every checkpoint that has never been gated,
    oldest first."""
    print(f"watching {cfg['candidate_dir']} every {cfg['poll_seconds']}s "
          f"(auto_promote={auto_promote}); champion {Path(cfg['champion_ck']).name}")
    while True:
        # re-read the config each pass: a promotion (or a hand edit) changes
        # the champion under us, and the next candidate must face the NEW one.
        cfg = load_config(cfg["path"])
        rows = read_history(cfg["history"])
        pend = pending_candidates(cfg["candidate_dir"], rows, cfg["champion_ck"])
        for p in pend:
            try:
                gate_one(cfg, str(p), auto_promote)
            except SchemaMismatchError as e:
                print(f"skipping {p}: {e}", file=sys.stderr)
                append_history(cfg["history"], history_row(
                    str(p), cfg["champion_ck"], 0, 0, FAIL, False,
                    f"schema mismatch: {e}", manual=False,
                    threshold=cfg["promote_threshold"], share=None))
                continue
            cfg = load_config(cfg["path"])
        _print_alert(cfg, read_history(cfg["history"]))
        if once:
            return 0
        sleep(cfg["poll_seconds"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="Champion gating: a candidate must beat the incumbent to be promoted.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="current champion, promotion history, pending candidates")

    g = sub.add_parser("gate", help="play CK vs champion (both side orders) and record a verdict")
    g.add_argument("ck")
    g.add_argument("--promote-if-pass", action="store_true",
                   help="move the champion pointer if the candidate passes (default: measure only)")

    w = sub.add_parser("watch", help="poll candidate_dir and gate new checkpoints")
    w.add_argument("--auto-promote", action="store_true")
    w.add_argument("--once", action="store_true", help="one pass, then exit")

    p = sub.add_parser("promote", help="MANUAL override: make CK the champion regardless of share")
    p.add_argument("ck")
    p.add_argument("--reason", required=True)

    r = sub.add_parser("reject", help="MANUAL override: record CK as rejected regardless of share")
    r.add_argument("ck")
    r.add_argument("--reason", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ChampionConfigError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    if args.cmd == "status":
        return cmd_status(cfg)
    if args.cmd == "gate":
        return cmd_gate(cfg, args.ck, args.promote_if_pass)
    if args.cmd == "watch":
        return cmd_watch(cfg, args.auto_promote, once=args.once)
    if args.cmd == "promote":
        return cmd_manual(cfg, args.ck, args.reason, promote=True)
    if args.cmd == "reject":
        return cmd_manual(cfg, args.ck, args.reason, promote=False)
    return 2


if __name__ == "__main__":
    sys.exit(main())
