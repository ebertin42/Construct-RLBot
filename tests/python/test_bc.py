"""B5: BC trainer on synthetic bc-export shards -- split stability, cached
class weights, deterministic batching, 2-epoch loss improvement, and
v1-schema checkpoint compatibility. This is the plan's binding test for B5."""
import json

import numpy as np
import torch

torch.set_num_threads(1)  # tiny nets; intra-op threads only fight the live trainer

from construct._engine import action_table_v1
from construct.learn.bc import (
    BCConfig,
    BCTrainer,
    class_counts,
    class_weights,
    iter_batches,
    rare_class_rows,
    split_shards,
)
from construct.learn.model_v1 import ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT, EntityPolicyNet

NUM_ACTIONS = 92


def _write_shard(path, n, rng):
    """Tiny synthetic bc-export shard, exact dtypes/shapes of bc_export.rs.
    Actions are drawn from a skewed distribution (class 16 dominates, like
    real replays' throttle-forward) so the inverse-frequency weights have
    something to do."""
    p = np.full(NUM_ACTIONS, 0.5 / (NUM_ACTIONS - 2))
    p[16], p[90] = 0.4, 0.1  # one common ground row, one "rare-ish" stall
    p /= p.sum()
    mask = np.zeros((n, MAX_ENT), dtype=np.uint8)
    mask[:, 1:6] = 1  # 1v1: mates/extra opps absent, like real exports
    np.savez(
        path,
        ents=rng.standard_normal((n, MAX_ENT, ENT_FEAT)).astype(np.float32),
        mask=mask,
        query=rng.standard_normal((n, Q_FEAT)).astype(np.float32),
        prev=rng.integers(0, NUM_ACTIONS, (n, PREV_ACTIONS)).astype(np.int64),
        action=rng.choice(NUM_ACTIONS, size=n, p=p).astype(np.int64),
    )


def _corpus(tmp_path, num_shards=6, n=160, seed=0):
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(num_shards):
        p = str(tmp_path / f"bc_{i:04d}.npz")
        _write_shard(p, n, rng)
        paths.append(p)
    return paths


def _cfg(tmp_path, **train_overrides):
    """bc_v1.toml shrunk to test size: tiny net, tiny batches, cpu."""
    train = dict(seed=0, lr=1e-3, batch_size=64, epochs=2, val_fraction=0.2,
                 weight_clamp=[0.1, 10.0], grad_clip=1.0)
    train.update(train_overrides)
    return BCConfig(
        data_dir=str(tmp_path),
        net={"d_model": 32, "layers": 1, "heads": 2, "ff": 64},
        train=train,
        run={"device": "cpu", "checkpoint_dir": str(tmp_path / "ck"),
             "log_every_batches": 1000},
    )


def test_config_toml_parses():
    cfg = BCConfig.load("configs/bc_v1.toml")
    assert cfg.data_dir == "data/bc"
    assert cfg.net == {"d_model": 128, "layers": 2, "heads": 4, "ff": 512}
    for key in ("seed", "lr", "weight_decay", "batch_size", "epochs",
                "val_fraction", "weight_clamp", "grad_clip"):
        assert key in cfg.train, key
    assert "device" in cfg.run


def test_split_is_stable_and_order_independent():
    paths = [f"/a/bc_{i:04d}.npz" for i in range(200)]
    train, val = split_shards(paths, 0.2)
    assert set(train) | set(val) == set(paths) and not set(train) & set(val)
    assert 20 <= len(val) <= 60  # ~40 expected; hash-split, not exact
    # order-independent: shuffling the input list changes nothing
    rng = np.random.default_rng(1)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    train2, val2 = split_shards(shuffled, 0.2)
    assert set(train2) == set(train) and set(val2) == set(val)
    # path-prefix-independent: only the FILENAME is hashed
    train3, val3 = split_shards([p.replace("/a/", "/elsewhere/") for p in paths], 0.2)
    assert {p.split("/")[-1] for p in val3} == {p.split("/")[-1] for p in val}


def test_class_weights_cached_and_reused(tmp_path):
    paths = _corpus(tmp_path, num_shards=3)
    cache = str(tmp_path / "bc_class_counts.json")
    out = class_counts(paths, NUM_ACTIONS, cache)
    counts = np.asarray(out["counts"])
    assert counts.sum() == 3 * 160
    assert sorted(out["shards"]) == [f"bc_{i:04d}.npz" for i in range(3)]

    w = class_weights(counts, (0.1, 10.0))
    assert w.dtype == np.float32 and w.shape == (NUM_ACTIONS,)
    assert (w >= 0.1).all() and (w <= 10.0).all()
    assert w[16] == w.min()  # most common class gets the smallest weight
    assert w[16] < w[90]     # rarer stall weighted above the common row

    # cache is REUSED, not recomputed: tamper with the counts (keeping the
    # shard set intact) and observe the tampered numbers come back
    with open(cache) as f:
        tampered = json.load(f)
    tampered["counts"] = [1] * NUM_ACTIONS
    with open(cache, "w") as f:
        json.dump(tampered, f)
    assert class_counts(paths, NUM_ACTIONS, cache)["counts"] == [1] * NUM_ACTIONS

    # stale cache (shard set changed) is recomputed
    extra = str(tmp_path / "bc_9999.npz")
    _write_shard(extra, 160, np.random.default_rng(9))
    out2 = class_counts(paths + [extra], NUM_ACTIONS, cache)
    assert np.asarray(out2["counts"]).sum() == 4 * 160

    # re-export that keeps a FILENAME but changes contents: byte size differs,
    # so the cache reads as stale and is recomputed (no silent stale counts)
    _write_shard(paths[0], 200, np.random.default_rng(10))
    out3 = class_counts(paths + [extra], NUM_ACTIONS, cache)
    assert np.asarray(out3["counts"]).sum() == 200 + 3 * 160


def test_fixed_seed_gives_identical_batches(tmp_path):
    paths = _corpus(tmp_path)
    a = list(iter_batches(paths, 64, seed=7, epoch=0))
    b = list(iter_batches(paths, 64, seed=7, epoch=0))
    assert len(a) == len(b)
    for ba, bb in zip(a, b):
        for k in ba:
            np.testing.assert_array_equal(ba[k], bb[k])
    # every batch full except possibly the last; nothing dropped
    assert all(x["action"].shape[0] == 64 for x in a[:-1])
    assert sum(x["action"].shape[0] for x in a) == 6 * 160
    # a different epoch reshuffles
    c = list(iter_batches(paths, 64, seed=7, epoch=1))
    assert any(not np.array_equal(x["action"], y["action"]) for x, y in zip(a, c))


def test_loader_threads_do_not_change_batch_stream(tmp_path):
    """The prefetcher's contract: parallel decompress, ordered consumption.
    loader_threads=1 (the old synchronous path) and loader_threads=4 must
    yield byte-identical batch sequences, shuffled and unshuffled."""
    paths = _corpus(tmp_path)
    for kwargs in ({"seed": 7, "epoch": 3}, {"shuffle": False}):
        a = list(iter_batches(paths, 64, loader_threads=1, **kwargs))
        b = list(iter_batches(paths, 64, loader_threads=4, **kwargs))
        assert len(a) == len(b)
        for ba, bb in zip(a, b):
            assert sorted(ba) == sorted(bb)
            for k in ba:
                np.testing.assert_array_equal(ba[k], bb[k])


def test_rare_class_rows_match_v1_table_semantics():
    rows = rare_class_rows(action_table_v1())
    assert list(rows["stall"]) == [90, 91]  # the two appended v1.1 stalls
    assert len(rows["jump"]) == 20          # 18 aerial jump rows + 2 stalls
    assert set(rows["stall"]) <= set(rows["jump"])


def test_two_epoch_run_improves_loss_and_checkpoint_is_v1_compatible(tmp_path):
    _corpus(tmp_path)
    cfg = _cfg(tmp_path)
    trainer = BCTrainer(cfg, action_table_v1())
    assert trainer.train_shards and trainer.val_shards
    value_before = [p.clone() for p in trainer.net.value_head.parameters()]

    history = trainer.run()
    assert len(history) == 2
    assert history[1]["train_loss"] < history[0]["train_loss"]
    for m in history:  # val metrics present and sane
        assert 0.0 <= m["top1"] <= m["top3"] <= 1.0
        assert np.isfinite(m["val_loss"])

    # value head untouched (BC trains the policy head only -- deferred per plan)
    for before, after in zip(value_before, trainer.net.value_head.parameters()):
        assert torch.equal(before, after)

    # checkpoint mirrors the v1 Trainer schema and loads back into the net
    ck_path = tmp_path / "ck" / "ck_bc_ep01.pt"
    assert ck_path.exists()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert ck["schema_version"] == 1
    assert ck["config"]["net"] == cfg.net
    assert ck["total_steps"] == 2 * sum(
        trainer.shard_samples[p.split("/")[-1]] for p in trainer.train_shards
    )
    net = EntityPolicyNet(
        d_model=cfg.net["d_model"], layers=cfg.net["layers"], heads=cfg.net["heads"],
        ff=cfg.net["ff"], action_table=ck["model"]["action_table"].numpy(),
    )
    net.load_state_dict(ck["model"])  # strict=True: exact key/shape match
    logits, value = net(
        torch.zeros(3, MAX_ENT, ENT_FEAT), torch.zeros(3, MAX_ENT, dtype=torch.bool),
        torch.zeros(3, Q_FEAT), torch.zeros(3, PREV_ACTIONS, dtype=torch.int64),
    )
    assert logits.shape == (3, NUM_ACTIONS) and value.shape == (3, 1)


def test_trainer_is_deterministic_across_runs(tmp_path):
    _corpus(tmp_path)
    cfg = _cfg(tmp_path, epochs=1)
    h1 = BCTrainer(cfg, action_table_v1()).run()
    h2 = BCTrainer(cfg, action_table_v1()).run()
    assert h1[0]["train_loss"] == h2[0]["train_loss"]
    assert h1[0]["val_loss"] == h2[0]["val_loss"]
