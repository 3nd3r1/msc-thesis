"""proxy+switch over a grid of decay and proxy head start, against the proxy alone."""

import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT / "experiments/03_link_guided_search"))

from run import DATASETS, RECALLS, loaded, proxy_scores, subgraph
from search import DECAY, adjacency, search

RESULTS = Path(__file__).resolve().parent / "results"
SEEDS = 20
DECAYS = [0.9, 0.97, 0.99, 0.999, 1.0]
HEADS = [0, 3, 10, 30]
NAMES = ["stack", "stack_author", "cora", "cad", "cad_thread", "cad_author"]

SHARED = {}


def one(job):
    k, decay, head, seed = job
    labels, proxy = SHARED["preds"][k]
    order = "proxy" if decay is None else "proxy+switch"
    hits = search(labels, order, None, SHARED["neighbours"], seed, RECALLS, proxy,
                  decay or DECAY, head)
    return job, 100 * np.array(hits) / len(labels)


def gaps(name):
    """switch minus proxy in points of the rows, per setting, predicate and target."""
    ds = DATASETS[name]
    nodes, edges, _, masks, pools = loaded(ds)
    scores = proxy_scores(ds, len(nodes))
    mask = pools[1][1]
    SHARED["neighbours"] = adjacency(subgraph(mask, edges), int(mask.sum()))
    SHARED["preds"] = [(masks[k][mask], scores[k][mask]) for k, _, _ in ds.predicates]
    preds = range(len(ds.predicates))
    jobs = [(k, None, 0, 0) for k in preds] + [
        (k, d, h, s) for k, d, h, s in product(preds, DECAYS, HEADS, range(SEEDS))
    ]
    with ProcessPoolExecutor(mp_context=get_context("fork")) as pool:
        done = dict(pool.map(one, jobs, chunksize=4))
    return {
        (d, h): np.array([
            np.mean([done[(k, d, h, s)] for s in range(SEEDS)], axis=0) - done[(k, None, 0, 0)]
            for k in preds
        ])
        for d, h in product(DECAYS, HEADS)
    }


RESULTS.mkdir(exist_ok=True)
lines = [
    "proxy+switch minus proxy, in points of the rows, negative means switch needs fewer calls",
    f"over every predicate and target {', '.join(f'{r:.0%}' for r in RECALLS)}, "
    f"switch is the mean of {SEEDS} seeds",
    "pool with junk and repeated texts dropped",
    f"head is pseudo hits for the proxy at the start, the committed setting is decay {DECAY} head 0",
    "",
    f"{'dataset':>13} {'decay':>6} {'head':>5} {'mean':>6} {'worst':>6} {'best':>6} {'> +0.5':>7}",
]
for name in NAMES:
    for (d, h), g in gaps(name).items():
        lines.append(f"{name:>13} {d:>6} {h:>5} {g.mean():>+6.2f} {g.max():>+6.1f} "
                     f"{g.min():>+6.1f} {int((g > 0.5).sum()):>3}/{g.size}")
        print(lines[-1], flush=True)
    lines.append("")
(RESULTS / "sweep.txt").write_text("\n".join(lines) + "\n")
