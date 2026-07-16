"""Checkpoint registry + TrueSkill ladder (jsonl, atomic rewrites)."""
import json
import os
import time

import trueskill

# Seer protocol parameters
TS_ENV = trueskill.TrueSkill(mu=25.0, sigma=25 / 3, beta=25 / 6, tau=25 / 300,
                             draw_probability=0.02)


class Registry:
    def __init__(self, path="league/registry.jsonl"):
        self.path = path
        self._entries: list[dict] = []
        if os.path.exists(path):
            with open(path) as f:
                self._entries = [json.loads(line) for line in f if line.strip()]
            # Backward-compatible read of pre-v1 registries: lines written before
            # schema_version existed default to 0 (the only schema that existed
            # then). Normalized once at load time so every in-memory entry
            # (and every subsequent _save()) always carries the field.
            for e in self._entries:
                e.setdefault("schema_version", 0)

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            for e in self._entries:
                f.write(json.dumps(e) + "\n")
        os.replace(tmp, self.path)

    def _find(self, ck):
        for e in self._entries:
            if e["ck"] == ck:
                return e
        raise KeyError(ck)

    def add(self, ck, steps, run, reward_config, schema_version=0):
        if any(e["ck"] == ck for e in self._entries):
            return
        self._entries.append({
            "ck": ck, "steps": steps, "run": run, "reward_config": reward_config,
            "schema_version": schema_version,
            "added_ts": int(time.time()),
            "mu": TS_ENV.mu, "sigma": TS_ENV.sigma, "games": 0,
        })
        self._save()

    def entries(self, schema_version=None):
        if schema_version is None:
            return list(self._entries)
        return [e for e in self._entries if e["schema_version"] == schema_version]

    def rating(self, ck):
        e = self._find(ck)
        return e["mu"], e["sigma"]

    def record_match(self, ck_a, ck_b, goals_a, goals_b):
        ea, eb = self._find(ck_a), self._find(ck_b)
        ra = TS_ENV.create_rating(ea["mu"], ea["sigma"])
        rb = TS_ENV.create_rating(eb["mu"], eb["sigma"])
        if goals_a > goals_b:
            ra, rb = trueskill.rate_1vs1(ra, rb, env=TS_ENV)
        elif goals_b > goals_a:
            rb, ra = trueskill.rate_1vs1(rb, ra, env=TS_ENV)
        else:
            ra, rb = trueskill.rate_1vs1(ra, rb, drawn=True, env=TS_ENV)
        ea["mu"], ea["sigma"] = ra.mu, ra.sigma
        eb["mu"], eb["sigma"] = rb.mu, rb.sigma
        ea["games"] += 1
        eb["games"] += 1
        self._save()

    def ladder(self):
        out = []
        for e in self._entries:
            r = TS_ENV.create_rating(e["mu"], e["sigma"])
            out.append({**e, "skill": TS_ENV.expose(r)})
        out.sort(key=lambda e: e["skill"], reverse=True)
        return out
