"""Behavior-cloning (BC) pretrain on bc-export tensors (Task B5 of
docs/superpowers/plans/2026-07-17-bc-pretrain.md).

Consumes the `bc_*.npz` shards written by the Rust `bc-export` binary
(`ents [S,17,26] f32, mask [S,17] u8 (1 = masked), query [S,64] f32,
prev [S,5] i64, action [S] i64` -- indices into the 92-row v1.1 action
table) and trains EntityPolicyNet's POLICY HEAD with inverse-action-
frequency-weighted cross-entropy. The value head is deliberately left at
init (plan Global Constraints: value warm-start is deferred; the KL-PPO
stage treats the BC net as a frozen *prior*, not a critic). No code path
enforces that -- it falls out of the loss: cross-entropy never touches
`value_head`, so its params never receive a gradient and AdamW skips them.

Data streaming: npz archives are zip files, so numpy cannot mmap them --
each shard is loaded whole (a shard is a few tens of MB), with shuffled
shard order + an in-shard permutation per epoch, all driven by one seeded
rng (fixed seed => identical batch sequence, the repo's determinism
contract). Decompression is the loader's cost, and zlib inflate releases
the GIL, so `loader_threads` (train config, default 4) worker threads
decompress up to _MAX_INFLIGHT shards ahead while the training loop runs --
but shards are always CONSUMED in shard order, so the batch stream is
byte-identical for any thread count (see _iter_shards). Train/val split is
95/5 by a stable hash of the shard FILENAME (not list order), so the split
survives re-listing, resharding of other files, and machine moves.

Training batches also get anti-copycat prev-action dropout (train config
`prev_dropout`, default 0.5): per-sample Bernoulli zeroing of the whole
5-slot prev ring, applied by `apply_prev_dropout` AFTER batch assembly from
a dedicated rng seeded off (seed, epoch, 0xD0) -- a separate stream from
the shuffle/permutation rng above, so dropout never perturbs the shard
order or in-shard permutation, and `prev_dropout = 0.0` is a true no-op
(the rng is never even drawn from). It is applied only in `train_epoch`;
`evaluate()` and any other consumer of `iter_batches` see the real prev
ring unchanged, and obs_v1's layout is untouched -- deployed/eval-time
inference always gets real prev actions.

Class weights: one first pass over every shard's `action` array counts the
92 classes; counts are cached to json next to the data and recomputed only
if the cache is missing or its shard set (filenames AND byte sizes) no
longer matches the directory. Weights are total/(A*count), clamped to
[0.1, 10] (config).

Checkpoints mirror learn/train.py's Trainer.save_checkpoint schema exactly
(schema_version: 1, model/optimizer/total_steps/config.net/...) so
eval_metrics.py, watch.py, and the league/kickstart tooling load BC
checkpoints unchanged. `config.ppo`/`config.env` are present but empty:
resuming PPO directly from a BC checkpoint is not a supported path (seed
the KL-PPO stage instead), and resume_train.py fails loudly on the empty
env rather than silently training with wrong settings.

Reference targets (Seer / Rolv BC precedent): val top-1 of ~35-50% on the
92-way human-action prediction task is NORMAL -- watch per-class recall on
the rare classes (jump rows, stalls) rather than chasing top-1.

Prev-action dropout exists because of an empirical diagnosis (2026-07-16):
an epoch-1 net trained without it learned a copycat policy -- 79% of its
predictions equalled prev[0] (humans repeat their own last action ~70% of
the time), zeroing the prev ring at eval time collapsed val accuracy from
69.6% to 19.3%, and closed-loop rollout was barely better than random
(0.26 touches/min vs a 0.16 random-policy baseline). Randomly blinding the
net to prev during training (see above) forces it to read entity/query
game state instead of parroting its own last output.
"""

import glob
import hashlib
import itertools
import json
import math
import os
import time
import tomllib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from construct.learn.model_v1 import EntityPolicyNet

BC_KEYS = ("ents", "mask", "query", "prev", "action")


@dataclass
class BCConfig:
    data_dir: str
    net: dict = field(default_factory=dict)     # d_model/layers/heads/ff
    # seed, lr, weight_decay, batch_size, epochs, val_fraction,
    # weight_clamp [lo, hi], grad_clip
    train: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict)     # device, checkpoint_dir, log_every_batches

    @classmethod
    def load(cls, path: str) -> "BCConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(**raw)


def find_shards(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "bc_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no bc_*.npz shards under {data_dir!r}")
    return paths


def _shard_unit_hash(path: str) -> float:
    """Stable [0,1) hash of the shard FILENAME (never the list order, never
    Python's salted hash()) -- the train/val split must not move between
    runs, re-listings, or machines."""
    name = os.path.basename(path)
    h = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def split_shards(paths: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """95/5-style split by shard-name hash. Order-independent and stable."""
    assert 0.0 <= val_fraction < 1.0, f"val_fraction must be in [0, 1), got {val_fraction}"
    train = [p for p in paths if _shard_unit_hash(p) >= val_fraction]
    val = [p for p in paths if _shard_unit_hash(p) < val_fraction]
    return train, val


def class_counts(paths: list[str], num_actions: int, cache_path: str) -> dict:
    """First pass over every shard's `action` array -> per-class counts,
    cached as json next to the data. Recomputed only when the cache is
    missing, was built for a different action count, or its shard set --
    filenames AND byte sizes -- no longer matches `paths` (a stale cache
    after a re-export, even one keeping the same filenames, must not skew
    the weights silently)."""
    sizes = {os.path.basename(p): os.path.getsize(p) for p in paths}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        # older caches stored a bare sample count per shard -- no size to
        # compare, so they read as stale and recompute once
        cached_sizes = {k: v.get("bytes") if isinstance(v, dict) else None
                        for k, v in cached.get("shards", {}).items()}
        if cached.get("num_actions") == num_actions and cached_sizes == sizes:
            return cached
        print(f"bc: class-count cache {cache_path} is stale, recomputing", flush=True)

    counts = np.zeros(num_actions, dtype=np.int64)
    shards: dict[str, dict] = {}
    for p in paths:
        name = os.path.basename(p)
        with np.load(p) as z:
            a = z["action"]
        assert a.min() >= 0 and a.max() < num_actions, (
            f"{p}: action indices outside [0, {num_actions})"
        )
        counts += np.bincount(a, minlength=num_actions)
        shards[name] = {"samples": int(a.shape[0]), "bytes": sizes[name]}
    out = {"num_actions": num_actions, "counts": counts.tolist(), "shards": shards}
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, cache_path)
    return out


def class_weights(counts: np.ndarray, clamp: tuple[float, float] = (0.1, 10.0)) -> np.ndarray:
    """Inverse-frequency CE weights: total/(A*count), so a uniform action
    distribution gives every class weight 1.0. Never-seen classes hit the
    upper clamp (they'd otherwise divide by zero -- and can't appear as
    targets anyway, so their weight only matters for cache reuse)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    w = total / (len(counts) * np.maximum(counts, 1.0))
    return np.clip(w, clamp[0], clamp[1]).astype(np.float32)


def rare_class_rows(action_table: np.ndarray) -> dict[str, np.ndarray]:
    """Rare-class groups for per-class recall, derived from the 92-table's
    row semantics (engine/src/actions.rs make_lookup_table_v1, row layout
    [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]): `jump` =
    rows that press jump; `stall` = the two appended v1.1 stall rows -- the
    only jump rows with nonzero yaw (the aerial loop skips jump & yaw!=0)."""
    t = np.asarray(action_table)
    jump = t[:, 5] != 0.0
    stall = jump & (t[:, 3] != 0.0)
    return {"jump": np.flatnonzero(jump), "stall": np.flatnonzero(stall)}


def _load_shard(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        missing = [k for k in BC_KEYS if k not in z.files]
        assert not missing, f"{path}: missing arrays {missing} (not a bc-export shard?)"
        return {k: z[k] for k in BC_KEYS}


# Bounded prefetch window: at most this many decompressed shards in flight
# (submitted but not yet consumed), ~8 x ~25 MB decompressed on the real
# corpus. Bounds memory no matter how far the loaders outrun the GPU.
_MAX_INFLIGHT = 8


def _iter_shards(order: list[str], loader_threads: int):
    """Yield decompressed shard dicts for `order`, strictly IN ORDER. With
    loader_threads > 1 a thread pool decompresses up to _MAX_INFLIGHT shards
    ahead (np.load's zlib inflate releases the GIL, so workers overlap with
    the training loop), but results are consumed in submission order --
    parallel decompress, ordered consumption -- so the caller sees the exact
    shard sequence the synchronous path produces. loader_threads <= 1 is the
    plain synchronous loop."""
    if loader_threads <= 1:
        for p in order:
            yield _load_shard(p)
        return
    with ThreadPoolExecutor(max_workers=loader_threads,
                            thread_name_prefix="bc-loader") as pool:
        it = iter(order)
        pending = deque(pool.submit(_load_shard, p)
                        for p in itertools.islice(it, _MAX_INFLIGHT))
        while pending:
            arrs = pending.popleft().result()  # re-raises worker exceptions
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(_load_shard, nxt))
            yield arrs


def iter_batches(paths, batch_size: int, *, seed: int = 0, epoch: int = 0,
                 shuffle: bool = True, loader_threads: int = 4):
    """Yield {ents, mask, query, prev, action} numpy batches. Shuffled shard
    order + in-shard permutation from one rng seeded by (seed, epoch), so a
    fixed seed replays the identical batch sequence and each epoch sees a
    different order. A carry buffer stitches shard remainders together, so
    every batch is full-size except possibly the last of the epoch.

    loader_threads only controls how far ahead shard DECOMPRESSION runs
    (_iter_shards); shards are consumed -- and the rng advanced -- in shard
    order, so the batch stream is byte-identical for any thread count."""
    order = list(paths)
    rng = np.random.default_rng([seed, epoch])
    if shuffle:
        rng.shuffle(order)
    carry: dict[str, np.ndarray] | None = None
    for arrs in _iter_shards(order, loader_threads):
        if shuffle:
            perm = rng.permutation(arrs["action"].shape[0])
            arrs = {k: v[perm] for k, v in arrs.items()}
        if carry is not None:
            arrs = {k: np.concatenate([carry[k], arrs[k]]) for k in BC_KEYS}
        n = arrs["action"].shape[0]
        n_full = n - n % batch_size
        for i in range(0, n_full, batch_size):
            yield {k: v[i:i + batch_size] for k, v in arrs.items()}
        carry = {k: v[n_full:] for k, v in arrs.items()} if n_full < n else None
    if carry is not None:
        yield carry


def batch_to_tensors(batch: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    """Numpy batch -> the tensor dict EntityPolicyNet.forward expects. mask
    arrives as u8 (1 = absent/masked, bc_obs.rs) and the net wants bool
    True = masked -- same convention, just a dtype cast."""
    return {
        "ents": torch.as_tensor(batch["ents"], device=device),
        "mask": torch.as_tensor(batch["mask"].astype(bool), device=device),
        "query": torch.as_tensor(batch["query"], device=device),
        "prev": torch.as_tensor(batch["prev"], device=device),
    }


def apply_prev_dropout(batch: dict[str, np.ndarray], p: float,
                       rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Anti-copycat regularizer (see module docstring): per-sample
    Bernoulli(p) zeroing of the whole 5-slot prev-action ring. Training
    only -- callers must not use this on val/evaluate batches.

    p <= 0.0 is a TRUE no-op: `batch` is returned unchanged and `rng` is
    never drawn from, so a p=0.0 training run's batch stream is byte-
    identical to code with no dropout at all. Samples that draw "zero" get
    a fresh `prev` array (the original batch/shard array is never mutated
    in place); samples that don't are untouched.
    """
    if p <= 0.0:
        return batch
    n = batch["action"].shape[0]
    zero = rng.random(n) < p
    if not zero.any():
        return batch
    prev = batch["prev"].copy()
    prev[zero] = 0
    return {**batch, "prev": prev}


class BCTrainer:
    def __init__(self, cfg: BCConfig, action_table: np.ndarray):
        self.cfg = cfg
        t = cfg.train
        self.seed = int(t.get("seed", 0))
        self.batch_size = int(t.get("batch_size", 4096))
        self.epochs = int(t.get("epochs", 4))
        self.grad_clip = float(t.get("grad_clip", 1.0))
        # per-sample prob of zeroing the whole prev ring during TRAINING
        # only (apply_prev_dropout / train_epoch) -- anti-copycat, see
        # module docstring. 0.0 disables (byte-identical batch stream).
        self.prev_dropout = float(t.get("prev_dropout", 0.5))
        assert 0.0 <= self.prev_dropout <= 1.0, (
            f"prev_dropout must be in [0, 1], got {self.prev_dropout}"
        )
        # shard-decompression threads; 1 = synchronous (the old behavior).
        # Any value yields the identical batch stream -- see iter_batches.
        self.loader_threads = max(1, int(t.get("loader_threads", 4)))
        clamp = tuple(t.get("weight_clamp", (0.1, 10.0)))

        self.shards = find_shards(cfg.data_dir)
        self.train_shards, self.val_shards = split_shards(
            self.shards, float(t.get("val_fraction", 0.05))
        )
        assert self.train_shards, "hash split left zero train shards"
        if not self.val_shards:
            print("bc: hash split left zero val shards (tiny corpus?) -- "
                  "val metrics will be skipped", flush=True)

        num_actions = int(np.asarray(action_table).shape[0])
        # Counts run over train+val shards INTENTIONALLY: class weights are
        # corpus statistics, and the cache stays valid when val_fraction moves.
        cache = class_counts(
            self.shards, num_actions,
            os.path.join(cfg.data_dir, "bc_class_counts.json"),
        )
        self.counts = np.asarray(cache["counts"], dtype=np.int64)
        self.shard_samples = {k: v["samples"] for k, v in cache["shards"].items()}
        weights = class_weights(self.counts, clamp)

        dev = cfg.run.get("device", "cuda")
        self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
        torch.manual_seed(self.seed)
        self.net = EntityPolicyNet(
            d_model=int(cfg.net["d_model"]), layers=int(cfg.net["layers"]),
            heads=int(cfg.net["heads"]), ff=int(cfg.net["ff"]),
            action_table=action_table,
        ).to(self.device)
        self.weights = torch.as_tensor(weights, device=self.device)
        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=float(t.get("lr", 3e-4)),
            weight_decay=float(t.get("weight_decay", 0.01)),
        )
        # Cosine anneal over the whole planned run. Batches per epoch is exact:
        # the carry buffer in iter_batches makes it ceil(train samples / batch).
        train_samples = sum(self.shard_samples[os.path.basename(p)] for p in self.train_shards)
        self.batches_per_epoch = math.ceil(train_samples / self.batch_size)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max(1, self.epochs * self.batches_per_epoch)
        )
        self.rare = rare_class_rows(action_table)
        self.total_steps = 0  # samples seen, mirrors Trainer's transition count

    def _loss(self, batch: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        obs = batch_to_tensors(batch, self.device)
        target = torch.as_tensor(batch["action"], device=self.device)
        logits, _value = self.net(**obs)  # value head unused: stays at init
        return F.cross_entropy(logits, target, weight=self.weights), logits

    def train_epoch(self, epoch: int) -> float:
        self.net.train()
        log_every = int(self.cfg.run.get("log_every_batches", 50))
        # Dedicated stream for prev-action dropout, seeded off (seed, epoch)
        # like the shuffle rng but salted so it never shares state with it --
        # drawing from it must not perturb iter_batches' shard/permutation
        # sequence (see apply_prev_dropout / module docstring).
        dropout_rng = np.random.default_rng((self.seed, epoch, 0xD0))
        loss_sum = n_sum = 0
        t_last, n_last = time.perf_counter(), 0
        for i, batch in enumerate(iter_batches(
                self.train_shards, self.batch_size, seed=self.seed, epoch=epoch,
                loader_threads=self.loader_threads)):
            batch = apply_prev_dropout(batch, self.prev_dropout, dropout_rng)
            loss, _ = self._loss(batch)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.opt.step()
            self.sched.step()
            n = batch["action"].shape[0]
            # weighted CE normalizes each batch by its weight sum; aggregating
            # by sample count is a known approximation (logging only -- the
            # same holds for evaluate()'s val_loss; gradients are unaffected)
            loss_sum += loss.item() * n
            n_sum += n
            self.total_steps += n
            if (i + 1) % log_every == 0:
                now = time.perf_counter()
                sps = (n_sum - n_last) / max(now - t_last, 1e-9)
                t_last, n_last = now, n_sum
                print(f"bc epoch {epoch} batch {i + 1}/{self.batches_per_epoch} "
                      f"loss {loss_sum / n_sum:.4f} lr {self.sched.get_last_lr()[0]:.2e} "
                      f"{sps:.0f} samples/s",
                      flush=True)
        return loss_sum / max(1, n_sum)

    @torch.no_grad()
    def evaluate(self) -> dict | None:
        """Val loss (same weighted CE as train, comparable numbers), top-1/
        top-3 accuracy, and macro recall over the rare-class groups (only
        classes with val support count toward a group's mean)."""
        if not self.val_shards:
            return None
        self.net.eval()
        A = self.weights.shape[0]
        loss_sum = n_sum = top1 = top3 = 0
        hits = np.zeros(A, dtype=np.int64)
        support = np.zeros(A, dtype=np.int64)
        for batch in iter_batches(self.val_shards, self.batch_size, shuffle=False,
                                  loader_threads=self.loader_threads):
            loss, logits = self._loss(batch)
            target = torch.as_tensor(batch["action"], device=self.device)
            n = target.shape[0]
            loss_sum += loss.item() * n
            n_sum += n
            pred = logits.argmax(-1)
            top1 += int((pred == target).sum())
            top3 += int((logits.topk(3, dim=-1).indices == target[:, None]).any(-1).sum())
            y = batch["action"]
            p = pred.cpu().numpy()
            support += np.bincount(y, minlength=A)
            hits += np.bincount(y[p == y], minlength=A)
        self.net.train()
        out = {"val_loss": loss_sum / n_sum, "top1": top1 / n_sum, "top3": top3 / n_sum}
        for name, rows in self.rare.items():
            seen = rows[support[rows] > 0]
            recall = float((hits[seen] / support[seen]).mean()) if len(seen) else float("nan")
            out[f"recall_{name}"] = recall
            out[f"support_{name}"] = int(support[rows].sum())
        return out

    def run(self) -> list[dict]:
        ck_dir = self.cfg.run.get("checkpoint_dir")
        print(f"bc: {len(self.train_shards)} train / {len(self.val_shards)} val shards, "
              f"{self.batches_per_epoch} batches/epoch x {self.epochs} epochs "
              f"(device {self.device})", flush=True)
        history = []
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            metrics = {"epoch": epoch, "train_loss": train_loss}
            val = self.evaluate()
            msg = f"bc epoch {epoch} done: train_loss {train_loss:.4f}"
            if val is not None:
                metrics.update(val)
                msg += (f" val_loss {val['val_loss']:.4f} top1 {val['top1']:.3f} "
                        f"top3 {val['top3']:.3f} recall_jump {val['recall_jump']:.3f} "
                        f"recall_stall {val['recall_stall']:.3f}")
            print(msg, flush=True)
            if ck_dir:
                os.makedirs(ck_dir, exist_ok=True)
                self.save_checkpoint(os.path.join(ck_dir, f"ck_bc_ep{epoch:02d}.pt"), metrics)
            history.append(metrics)
        return history

    def save_checkpoint(self, path: str, metrics: dict | None = None):
        """Mirrors learn/train.py Trainer.save_checkpoint key-for-key so v1
        tooling (eval_metrics.py, watch.py, league load_sd) reads BC
        checkpoints unchanged. ppo/env are empty by design -- see module
        docstring. The extra `bc` key is provenance only."""
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.opt.state_dict(),
                "total_steps": self.total_steps,
                "schema_version": 1,
                "config": {"net": self.cfg.net, "ppo": {}, "env": {}},
                "reward_config_path": "",
                "curriculum_config_path": "",
                "bc": {"data_dir": self.cfg.data_dir, "train": self.cfg.train,
                       "metrics": metrics or {}},
            },
            path,
        )
