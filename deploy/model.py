import numpy as np
import torch
import torch.nn as nn

from actions import make_lookup_table_v1


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int, action_count: int, hidden: tuple[int, ...] = (512, 512)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_size
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, action_count)
        self.value_head = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


# --------------------------------------------------------------------------
# schema v1: entity-transformer net, vendored from
# python/construct/learn/model_v1.py — deploy must stay standalone on
# Windows, so no construct import here. Module names/shapes are the
# state_dict contract and are copied 1:1; training-only helpers
# (act/evaluate/aux_outputs, the param-count print) are dropped. Keep in
# sync manually if model_v1.py changes.
# --------------------------------------------------------------------------

ENT_FEAT = 26
Q_FEAT = 64
MAX_ENT = 17
PREV_ACTIONS = 5
ACT_EMB = 32


class MHA(nn.Module):
    """Multi-head attention from primitives (see model_v1.py).

    forward(q_in [B,Tq,d], kv_in [B,Tk,d], key_mask [B,Tk] bool True=masked)
    -> [B,Tq,d]
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0, f"d_model={d} not divisible by heads={heads}"
        self.d = d
        self.heads = heads
        self.dh = d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        B, Tq, _ = q_in.shape
        Tk = kv_in.shape[1]
        q = self.q(q_in).view(B, Tq, self.heads, self.dh).transpose(1, 2)
        k = self.k(kv_in).view(B, Tk, self.heads, self.dh).transpose(1, 2)
        v = self.v(kv_in).view(B, Tk, self.heads, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)  # [B,H,Tq,Tk]
        mask = key_mask[:, None, None, :]  # [B,1,1,Tk]
        scores = scores.masked_fill(mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # [B,H,Tq,dh]
        out = out.transpose(1, 2).reshape(B, Tq, self.d)
        return self.o(out)


class Block(nn.Module):
    """Pre-LN transformer block: self-attention among unmasked entities."""

    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MHA(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, mask)
        h2 = self.ln2(x)
        x = x + self.ff2(torch.relu(self.ff1(h2)))
        return x


class EntityPolicyNet(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        layers: int = 2,
        heads: int = 4,
        ff: int = 512,
        action_table: np.ndarray = None,
        aux: bool = False,
    ):
        super().__init__()
        assert action_table is not None, "action_table [N,8] is required"
        action_table = np.asarray(action_table, dtype=np.float32)
        assert action_table.ndim == 2 and action_table.shape[1] == 8, (
            f"action_table must be [N,8], got {action_table.shape}"
        )
        self.d_model = d_model
        self.aux = aux

        self.embed = nn.Linear(ENT_FEAT, d_model)
        self.query_embed = nn.Linear(Q_FEAT, d_model)
        self.act_embed = nn.Sequential(
            nn.Linear(8, d_model), nn.ReLU(), nn.Linear(d_model, ACT_EMB)
        )
        self.prev_embed_w = nn.Parameter(torch.zeros(PREV_ACTIONS))
        self.prev_proj = nn.Linear(ACT_EMB, d_model)

        self.blocks = nn.ModuleList([Block(d_model, heads, ff) for _ in range(layers)])

        self.pool_ln = nn.LayerNorm(d_model)
        self.pool = MHA(d_model, heads)

        self.policy_dot = nn.Linear(d_model, ACT_EMB)
        self.value_head = nn.Linear(d_model, 1)

        if aux:
            self.aux_reward = nn.Linear(d_model, 3)
            self.aux_recon = nn.Linear(d_model, MAX_ENT * ENT_FEAT)

        # non-trainable action lookup table; travels with the state_dict
        self.register_buffer("action_table", torch.as_tensor(action_table, dtype=torch.float32))

    def forward(
        self,
        ents: torch.Tensor,     # [B,17,26] f32
        mask: torch.Tensor,     # [B,17] bool True=ignore
        query: torch.Tensor,    # [B,64] f32
        prev: torch.Tensor,     # [B,5] int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(ents)  # [B,17,d]
        for blk in self.blocks:
            x = blk(x, mask)

        q = self.query_embed(query).unsqueeze(1)  # [B,1,d]
        pooled = self.pool(q, self.pool_ln(x), mask).squeeze(1)  # [B,d]

        prev_rows = self.action_table[prev]         # [B,5,8]
        prev_e = self.act_embed(prev_rows)           # [B,5,32]
        w = torch.softmax(self.prev_embed_w, dim=0)   # [5]
        prev_sum = (prev_e * w.view(1, PREV_ACTIONS, 1)).sum(dim=1)  # [B,32]
        pooled = pooled + self.prev_proj(prev_sum)

        player = self.policy_dot(pooled)              # [B,32]
        table_e = self.act_embed(self.action_table)    # [N,32]
        logits = player @ table_e.T                    # [B,N]

        value = self.value_head(pooled)  # [B,1]
        return logits, value


def load_policy(checkpoint_path: str) -> tuple[nn.Module, int]:
    """Loads checkpoint.pt and dispatches on its schema_version.

    Returns (net, schema_version):
      0 -> PolicyValueNet (obs 94, 90-action table)
      1 -> EntityPolicyNet (obs_v1 entity set, 92-action table)
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    version = int(ck["schema_version"])
    if version == 0:
        net = PolicyValueNet(94, 90, tuple(ck["config"]["net"]["hidden"]))
        net.load_state_dict(ck["model"])
    elif version == 1:
        cfg = ck["config"]["net"]
        table = make_lookup_table_v1()
        # fail loud if the checkpoint was trained against a different table
        # than the one deploy decodes with
        ck_table = ck["model"]["action_table"]
        assert ck_table.shape == tuple(table.shape) and torch.equal(
            ck_table, torch.as_tensor(table)
        ), "checkpoint action_table differs from deploy's make_lookup_table_v1()"
        net = EntityPolicyNet(
            d_model=cfg["d_model"], layers=cfg["layers"], heads=cfg["heads"], ff=cfg["ff"],
            action_table=table,
            aux=any(k.startswith("aux_") for k in ck["model"]),
        )
        net.load_state_dict(ck["model"])  # strict: exact key/shape match
    else:
        raise AssertionError(f"unsupported checkpoint schema_version: {version}")
    net.eval()
    torch.set_num_threads(1)
    return net, version
