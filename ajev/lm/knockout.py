"""Knockout scoring for option sets larger than the label table (no torch needed).

Options are split into groups of at most 26 and each group is scored on its own (round 1). The top
``per_group`` options of every group go to a final. An option's probability is
(final probability mass of its group) x (its probability within the group), so all K options sum to 1.
"""

from __future__ import annotations

import math


def split_groups(k: int, max_size: int = 26) -> list[list[int]]:
    """Split 0..k-1 into contiguous groups of near-equal size, each at most max_size."""
    n = math.ceil(k / max_size)
    base, extra = divmod(k, n)
    groups, start = [], 0
    for g in range(n):
        size = base + (1 if g < extra else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def log_softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    s = math.log(sum(math.exp(x - m) for x in xs)) + m
    return [x - s for x in xs]


def finalists(groups: list[list[int]], group_logits: list[list[float]], per_group: int = 2) -> list[int]:
    """Top per_group options of each group (original indices, in order)."""
    out = []
    for g, lg in zip(groups, group_logits):
        top = sorted(range(len(g)), key=lambda j: -lg[j])[:per_group]
        out += [g[j] for j in sorted(top)]
    return out


def combine(k: int, groups: list[list[int]], group_logits: list[list[float]], final_idx: list[int],
            final_logits: list[float]) -> list[float]:
    """Log-probabilities for all k options: log M_g + log p_g(o)."""
    final_lp = log_softmax(final_logits)
    group_of = {o: gi for gi, g in enumerate(groups) for o in g}
    mass = [0.0] * len(groups)
    for o, lp in zip(final_idx, final_lp):
        mass[group_of[o]] += math.exp(lp)
    out = [0.0] * k
    for gi, (g, lg) in enumerate(zip(groups, group_logits)):
        lp_g = log_softmax(lg)
        log_m = math.log(mass[gi]) if mass[gi] > 0 else -1e9
        for j, o in enumerate(g):
            out[o] = log_m + lp_g[j]
    return out
