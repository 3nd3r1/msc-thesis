"""The reviews for the replay. Active, enough positives, and citation reaches a positive."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT / "experiments/03_link_guided_search"))

from data import load_synergy, synergy_meta
from run import correlation
from search import adjacency

MIN_POSITIVES = 40


def picked():
    """(name, rows, positives, nbr+) per picked review, most positives first."""
    out = []
    for name, meta in synergy_meta().items():
        if not meta.get("active", True):
            continue
        nodes, edges = load_synergy(name)
        mask = nodes.included.to_numpy()
        if mask.sum() < MIN_POSITIVES:
            continue
        nbr = correlation(mask, adjacency(edges, len(nodes)))["nbr+"]
        if nbr > 0:
            out.append((name, len(nodes), int(mask.sum()), nbr))
    return sorted(out, key=lambda x: -x[2])


if __name__ == "__main__":
    print(f"{'review':>20} {'rows':>7} {'pos':>5} {'nbr+':>6}")
    for name, n, pos, nbr in picked():
        print(f"{name:>20} {n:>7,} {pos:>5} {nbr:>6.1%}")
