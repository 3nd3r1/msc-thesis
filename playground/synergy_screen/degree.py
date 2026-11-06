"""Citation degree of included against excluded papers on the replay reviews."""

from pathlib import Path

import numpy as np

import pick  # noqa: F401, puts experiments/lib and 03 on the path
from data import load_synergy
from report import report
from run import SYNERGY_REVIEWS

if __name__ == "__main__":
    out, save = report(Path(__file__).resolve().parent / "results", "degree")
    out("deg: mean citation degree inside the pool, iso: share with no edge")
    out("+ is included, - is excluded, all rows")
    out()
    out(f"{'review':>20} {'deg+':>6} {'deg-':>6} {'iso+':>6} {'iso-':>6}")
    for name in SYNERGY_REVIEWS:
        nodes, edges = load_synergy(name)
        y = nodes.included.values.astype(bool)
        deg = np.bincount(edges.ravel(), minlength=len(nodes))
        out(
            f"{name:>20} {deg[y].mean():6.2f} {deg[~y].mean():6.2f} "
            f"{(deg[y] == 0).mean():6.1%} {(deg[~y] == 0).mean():6.1%}"
        )
    save()
