"""The reviews for the replay. Active, enough positives, and citation reaches a positive."""

import tomllib

import numpy as np

from screen import REPO, citations, pool
from run import correlation
from search import adjacency

MIN_POSITIVES = 40


def active():
    datasets = tomllib.load(open(REPO.parent / "datasets.toml", "rb"))["datasets"]
    return {d["key"] for d in datasets if d.get("active", True)}


def picked():
    """(name, rows, positives, nbr+) per picked review, most positives first."""
    keys = active()
    out = []
    for path in REPO.glob("*/*_ids.csv"):
        name = path.stem.replace("_ids", "")
        if name not in keys:
            continue
        data = pool(path)
        mask = np.array([bool(r["included"]) for r in data])
        if mask.sum() < MIN_POSITIVES:
            continue
        nbr = correlation(mask, adjacency(citations(data), len(data)))["nbr+"]
        if nbr > 0:
            out.append((name, len(data), int(mask.sum()), nbr))
    return sorted(out, key=lambda x: -x[2])


if __name__ == "__main__":
    print(f"{'review':>20} {'rows':>7} {'pos':>5} {'nbr+':>6}")
    for name, n, pos, nbr in picked():
        print(f"{name:>20} {n:>7,} {pos:>5} {nbr:>6.1%}")
