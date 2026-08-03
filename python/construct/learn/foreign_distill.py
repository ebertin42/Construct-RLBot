"""On-policy distillation from a foreign teacher (nexto / immortal).

Distinct from `kickstart.py`, which distils from a frozen v0 MLP checkpoint and therefore
has the teacher's full logit vector to match. A foreign teacher is queried inside the engine
and returns only the ROW IT CHOSE (`collect()["teacher_actions"]`, see `Engine::set_teacher`),
because both backends are `argmax` -- `foreign.rs` does `self.table[argmax(&l)]`. So the
target is a hard label and the loss is a cross-entropy, not a KL between distributions.

TWO THINGS THIS MODULE EXISTS TO GET RIGHT
------------------------------------------
1. **Equivalence classes.** The v1-air action table has 104 rows but only 96 distinct
   control vectors: {56,94,100}, {57,95,101}, {92,98}, {93,99}, {96,102}, {97,103} are
   byte-identical triples/pairs. The engine's `index_of_controls` returns the FIRST match, so
   a teacher that picked row 100 is labelled 56. Charging the student for putting its mass on
   94 instead of 56 would be charging it for a distinction that does not exist in the
   environment -- a pure ln(3)=1.0986 penalty floor it can never pay down. We therefore
   logsumexp the student's logits within each class BEFORE the cross-entropy, so the student
   is scored on the control vector it emits, which is the only thing the arena observes.

2. **The -1 sentinel.** `index_of_controls` returns None -> -1 when the teacher's controls
   match no row (it acts in a continuous space; our table is a 104-row quantisation). Those
   rows carry NO usable target and are masked out. They must never be allowed to index the
   table: `action_table[-1]` silently resolves to row 103 (jump+boost+pitch) in torch, and
   there is no null row -- row 0 is a real action.
"""

import torch
import torch.nn.functional as F


def equivalence_classes(action_table) -> torch.Tensor:
    """Map each action-table row to a canonical class id in [0, K).

    Rows with byte-identical control vectors share a class. Returns a LongTensor of shape
    [A]; `int(out.max()) + 1` is K. Mirrors `actions::index_of_controls`'s first-match
    canonicalisation on the Rust side, so a label the engine emits is always the lowest row
    index of its class and `cls_of[label]` is that class.
    """
    tbl = torch.as_tensor(action_table, dtype=torch.float32)
    if tbl.ndim != 2:
        raise ValueError(f"action_table must be 2-D [A, 8], got shape {tuple(tbl.shape)}")
    # Exact equality is correct here and float tolerance would be WRONG: these rows are
    # written as literals and duplicated verbatim, never computed, so identical rows are
    # bit-identical. A tolerance would merge genuinely distinct near-neighbours.
    uniq, inverse = torch.unique(tbl, dim=0, return_inverse=True)
    # torch.unique orders by value, not by first appearance. Renumber so class ids follow
    # the order rows first appear, making class 0 = row 0 and keeping the mapping stable and
    # readable against the table itself.
    seen, remap = {}, torch.empty_like(inverse)
    for row in range(inverse.numel()):
        u = int(inverse[row])
        if u not in seen:
            seen[u] = len(seen)
        remap[row] = seen[u]
    return remap


def collapse_logits(logits: torch.Tensor, cls_of: torch.Tensor, n_classes: int) -> torch.Tensor:
    """logsumexp `logits` [B, A] within each equivalence class -> [B, K].

    The student's probability of EMITTING a control vector is the sum of its probabilities
    over every row carrying that vector, so the class logit is the logsumexp of the member
    logits. Done in a numerically stable two-pass form (subtract per-class max) rather than
    exp-then-scatter-add, which overflows on the logit ranges PPO actually produces.
    """
    b = logits.shape[0]
    idx = cls_of.unsqueeze(0).expand(b, -1)
    neg_inf = torch.finfo(logits.dtype).min
    mx = torch.full((b, n_classes), neg_inf, dtype=logits.dtype, device=logits.device)
    mx = mx.scatter_reduce(1, idx, logits, reduce="amax", include_self=True)
    shifted = torch.exp(logits - mx.gather(1, idx))
    summed = torch.zeros((b, n_classes), dtype=logits.dtype, device=logits.device)
    summed = summed.scatter_add(1, idx, shifted)
    return mx + torch.log(summed)


def foreign_distill_loss(student_logits, teacher_actions, cls_of, n_classes):
    """Cross-entropy of the student's class distribution against the teacher's chosen class.

    student_logits: [B, A] raw student logits over action-table rows.
    teacher_actions: [B] long, row index chosen by the teacher, or -1 for "no exact match".
    Returns (loss, info). `loss` is a 0-d tensor, the mean over UNMASKED rows only; it is
    exactly 0 (and grad-connected via a *0 no-op) when every row is masked, so a minibatch
    that happens to contain no usable label cannot produce a NaN or a detached graph.
    """
    ta = teacher_actions.long()
    keep = ta >= 0
    n_keep = int(keep.sum())
    frac = n_keep / max(1, ta.numel())
    cls_logits = collapse_logits(student_logits, cls_of, n_classes)
    if n_keep == 0:
        return student_logits.sum() * 0.0, {"fd_label_frac": frac, "fd_ce": 0.0}
    target = cls_of[ta[keep]]
    loss = F.cross_entropy(cls_logits[keep], target)
    with torch.no_grad():
        # Reported so a run can be read against the state-INDEPENDENT baseline on the same
        # frames. A cross-entropy alone is not interpretable; the marginal is the ruler.
        # NO top-1 agreement is computed here, deliberately: 0.642 top-1 alongside a
        # 0W/1D/639L record is how replay-BC misled this project once.
        counts = torch.bincount(target, minlength=n_classes).float()
        p = counts / counts.sum()
        marg = float(-torch.log(p[target].clamp_min(1e-12)).mean())
    return loss, {"fd_label_frac": frac, "fd_ce": loss.detach().item(), "fd_marginal": marg}
