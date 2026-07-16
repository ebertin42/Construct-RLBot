"""Obs v1 entity-transformer policy/value net (pre-LN encoder, dot action head).

Mirrors model.py's PolicyValueNet act/evaluate API so ppo.py plugs in
unchanged (only the collect-side obs plumbing differs between v0 and v1).

Submodule names are a HARD CONTRACT with the Rust candle port (Task T4),
which loads this net's exact state_dict key names 1:1 -- do not rename.
"""

import numpy as np
import torch
import torch.nn as nn

ENT_FEAT = 26
Q_FEAT = 64
MAX_ENT = 17
PREV_ACTIONS = 5
ACT_EMB = 32


class MHA(nn.Module):
    """Multi-head attention from primitives (not nn.MultiheadAttention --
    its packed in_proj weight layout complicates the candle port).

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

        # non-trainable: the action lookup table (rows = discrete actions,
        # cols = the 8 controller floats). Not a Parameter -- never gets a
        # gradient -- but travels with the module (device/dtype, state_dict).
        self.register_buffer("action_table", torch.as_tensor(action_table, dtype=torch.float32))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"EntityPolicyNet: {n_params:,} params (d_model={d_model} layers={layers} "
              f"heads={heads} ff={ff} aux={aux} action_table={action_table.shape[0]})")

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

    def aux_outputs(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Config-gated aux heads (Lucy-SKG): reward-prediction + entity
        reconstruction from the pooled trunk embedding. Unused by default."""
        assert self.aux, "aux_outputs() called but net was built with aux=False"
        return self.aux_reward(pooled), self.aux_recon(pooled)

    @torch.no_grad()
    def act(self, ents, mask, query, prev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(ents, mask, query, prev)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, ents, mask, query, prev, actions):
        logits, value = self(ents, mask, query, prev)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)
