"""Cluster bootstrap intervals on the 06 gaps, links minus proxy, kNN, rewired and wide BARGAIN, on CAD and MELD."""

import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))

from corpora import DATASETS, loaded, proxy_scores, subgraph, vectors_for
from guarantee import bargain, clusters, order, qbcb, remap, resample, rewire, stop_count
from report import report
from search import adjacency, knn

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["cad", "meld"]
T, DELTA, K = 0.9, 0.1, 40
R = stop_count(K, T, DELTA)
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RUNS = 1000
BARGAIN_RUNS = 20
OTHERS = ["proxy", "knn", "rewired", "bargain"]
DATA = {}


def budgets():
    """The budget of the wide BARGAIN best per predicate, as a share of the rows or a count."""
    out = {}
    for line in (ROOT / "playground/bargain_screen/results/wide.txt").read_text().splitlines():
        m = re.match(r"\s*(cad|meld)\s+(.+?)\s+[\d.]+%\s+[\d.]+%\s+(\S+)\s+\d+%", line)
        if m:
            out[m.group(1), m.group(2)] = float(m.group(3))
    return out


def gaps(name, rows, copy, seed):
    """Mean over predicates of links minus each other arm on one set of rows, in points of the rows."""
    d = DATA[name]
    n = len(rows)
    links = remap(d["links"], rows, copy)
    graphs = {
        "proxy": None,
        "knn": adjacency(remap(d["knn"], rows, copy, within=False), n),
        "rewired": adjacency(rewire(links, seed), n),
        "links": adjacency(links, n),
    }
    out = {c: [] for c in OTHERS}
    for key, (y, p) in d["predicates"].items():
        y, p = y[rows], p[rows]
        cost = {}
        for arm, nb in graphs.items():
            o = order(p, y, nb)
            rng = np.random.default_rng(seed)
            cost[arm] = np.mean([qbcb(o, y, K, R, T, rng)[0] for _ in range(RUNS)])
        b = d["budgets"][key]
        budget = int(b) if b >= 1 else max(1, round(b * n))
        cost["bargain"] = np.mean([bargain(p, y, budget, T, DELTA, seed * 100 + s)[0] for s in range(BARGAIN_RUNS)])
        for c in OTHERS:
            out[c].append(100 * (cost["links"] - cost[c]))
    return {c: np.mean(v) for c, v in out.items()}


def one(job):
    name, draw = job
    rows, copy = resample(DATA[name]["group"], np.random.default_rng(draw))
    res = gaps(name, rows, copy, draw)
    print(name, draw, flush=True)
    return name, res


def main():
    b = budgets()
    for name in NAMES:
        ds = DATASETS[name]
        nodes, edges, _, masks, pools = loaded(ds)
        mask = pools[1][1]
        n = int(mask.sum())
        links = subgraph(mask, edges)
        scores = proxy_scores(ds, len(nodes))
        tie = np.random.default_rng(0).random(n) * 1e-9
        DATA[name] = {
            "links": links,
            "knn": knn(vectors_for(nodes)[mask], max(1, round(len(links) / n))),
            "group": clusters(links, n),
            "budgets": {key: b[name, key] for key, _, _ in ds.predicates},
            "predicates": {key: (masks[key][mask], np.nan_to_num(scores[key][mask]) + tie) for key, _, _ in ds.predicates},
        }

    full = {name: gaps(name, np.arange(len(DATA[name]["group"])), np.zeros(len(DATA[name]["group"]), int), 0) for name in NAMES}
    with ProcessPoolExecutor(mp_context=get_context("fork")) as pool:
        res = list(pool.map(one, [(name, d) for name in NAMES for d in range(1, DRAWS + 1)]))

    out, save = report(RESULTS, "gaps")
    out(f"cluster bootstrap over the link components, threads on CAD and dialogues on MELD, {DRAWS} draws")
    out(f"each draw reruns the 06 orders under QBCB with k {K} and r {R}, {RUNS} runs, and BARGAIN plus checking")
    out(f"at the budget of its wide best from playground/bargain_screen/results/wide.txt, {BARGAIN_RUNS} runs")
    out("kNN edges between clusters go to the first copy of the neighbour, rewired is redone on each draw's links")
    out("links minus each arm, mean over predicates, in points of the rows, full data then the 95% interval")
    out()
    out(f"{'dataset':>8} " + " ".join(f"{c:>22}" for c in OTHERS))
    for name in NAMES:
        draws = {c: np.array([r[c] for nm, r in res if nm == name]) for c in OTHERS}
        cells = [f"{full[name][c]:+.2f} [{np.percentile(draws[c], 2.5):+.2f}, {np.percentile(draws[c], 97.5):+.2f}]" for c in OTHERS]
        out(f"{name:>8} " + " ".join(f"{c:>22}" for c in cells))
    save()


if __name__ == "__main__":
    main()
