"""Assign whole groups to train/val/test so each split gets its share of every stratum.

A stratum is a kind of image, e.g. ("Real_Fire/Smoke", "smoke") or ("", "NEG"). Groups
are indivisible, so ratios come out close rather than exact. Greedy: biggest groups
first, each placed in the split where it most reduces the squared distance from the
targets, both overall and per stratum (normalised so rare strata still count).
"""

from __future__ import annotations

import random
from collections import Counter

from kiln.data.common import SPLITS


def assign(units: dict[str, Counter], ratios: tuple[int, int, int], seed: int = 0) -> dict[str, str]:
    """units: group name -> Counter(stratum -> images). Returns group name -> split."""
    share = {s: r / 100 for s, r in zip(SPLITS, ratios)}
    active = [s for s in SPLITS if share[s] > 0]
    totals: Counter = Counter()
    for u in units.values():
        totals.update(u)
    n_all = sum(totals.values())
    target = {s: {k: share[s] * v for k, v in totals.items()} for s in SPLITS}
    target_n = {s: share[s] * n_all for s in SPLITS}

    have = {s: Counter() for s in SPLITS}
    have_n = Counter()
    order = list(units)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -sum(units[g].values()))

    result = {}
    for g in order:
        u, size = units[g], sum(units[g].values())

        def cost(s: str) -> float:
            c = sum(((have[s][k] + v - target[s][k]) ** 2 - (have[s][k] - target[s][k]) ** 2) / totals[k]
                    for k, v in u.items())
            c += ((have_n[s] + size - target_n[s]) ** 2 - (have_n[s] - target_n[s]) ** 2) / n_all
            return c

        best = min(active, key=lambda s: (cost(s), have_n[s] - target_n[s]))
        result[g] = best
        have[best].update(u)
        have_n[best] += size
    return result
