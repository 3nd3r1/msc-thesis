"""Do copies of a row in a bootstrap draw explain why the 07 draws show a smaller link gain than the full data?

Each draw takes the same number of clusters without and with replacement, so the two differ only in the copies.
The number is 1 - 1/e of the clusters, the expected share of distinct clusters in a 07 draw.
Rule fixed before the run: copies explain the shrink on MELD if for every arm the mean gap with copies minus
the mean gap without copies is at least half of that arm's 07 shrink, the mean of the 07 draws minus the full data gap.
"""

import json
import os
import sys

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))

from corpora import DATASETS, loaded, proxy_scores, subgraph, vectors_for
from guarantee import cal, clusters, knn_draw, qbcb, remap, rewire, stop_count
from report import report
from search import adjacency

RESULTS = Path(__file__).resolve().parent / "results"
SEVEN = ROOT / "experiments/07_cal_links/results/draws.jsonl"
NAMES = ["cad", "meld"]
T, DELTA, K = 0.9, 0.1, 40
R = stop_count(K, T, DELTA)
DRAWS, RUNS = 100, 1000
ARMS = ["cal", "cal+knn", "cal+rewired", "cal+links"]
RULE = ["cal", "cal+knn", "cal+rewired"]
WORKERS = int(os.environ.get("WORKERS", max(1, (os.cpu_count() or 2) // 2)))
DONE = RESULTS / "copies_draws.jsonl"
DATA = {}


def subsample(group, rng, replace):
    """1 - 1/e of the clusters, drawn with or without replacement, as rows and which draw each came from."""
    by = np.argsort(group, kind="stable")
    members = np.split(by, np.cumsum(np.bincount(group))[:-1])
    m = round((1 - np.exp(-1)) * len(members))
    draw = rng.choice(len(members), m, replace=replace)
    rows = np.concatenate([members[g] for g in draw])
    copy = np.repeat(np.arange(m), [len(members[g]) for g in draw])
    return rows, copy


def costs(name, rows, copy, seed):
    d = DATA[name]
    n = len(rows)
    links = remap(d["links"], rows, copy)
    graphs = {
        "cal": None,
        "cal+knn": adjacency(knn_draw(d["vectors"], rows, d["k"]), n),
        "cal+rewired": adjacency(rewire(links, seed), n),
        "cal+links": adjacency(links, n),
    }
    X = d["vectors"][rows]
    out = {}
    for key, (y, p) in d["predicates"].items():
        y, p = y[rows], p[rows]
        row = {}
        for arm, nb in graphs.items():
            o = cal(p, y, X, nb)
            rng = np.random.default_rng(seed)
            row[arm] = 100 * np.mean([qbcb(o, y, K, R, T, rng)[0] for _ in range(RUNS)])
        out[key] = row
    return out


def one(job):
    name, draw, replace = job
    rows, copy = subsample(DATA[name]["group"], np.random.default_rng(draw), replace)
    print(name, draw, replace, flush=True)
    return name, draw, replace, costs(name, rows, copy, draw)


def gap(res, c):
    return float(np.mean([r["cal+links"] - r[c] for r in res.values()]))


def main():
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
            "vectors": vectors_for(nodes)[mask],
            "k": max(1, round(len(links) / n)),
            "group": clusters(links, n),
            "predicates": {key: (masks[key][mask], np.nan_to_num(scores[key][mask]) + tie) for key, _, _ in ds.predicates},
        }

    done = [json.loads(line) for line in DONE.read_text().splitlines()] if DONE.exists() else []
    seen = {(r["name"], r["draw"], r["replace"]) for r in done}
    jobs = [(name, d, rep) for d in range(1, DRAWS + 1) for name in NAMES for rep in (False, True)]
    jobs = [j for j in jobs if j not in seen]
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("fork")) as pool, DONE.open("a") as f:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            name, draw, replace, res = fut.result()
            f.write(json.dumps({"name": name, "draw": draw, "replace": replace, "costs": res}) + "\n")
            f.flush()
    done = [json.loads(line) for line in DONE.read_text().splitlines()]
    seven = [json.loads(line) for line in SEVEN.read_text().splitlines()]

    out, save = report(RESULTS, "copies")
    out(f"cal+links minus each arm, mean over predicates, in points of the rows, QBCB with k {K} and target {T}")
    out("07 full and 07 draws from experiments/07_cal_links/results/draws.jsonl")
    out(f"without and with copies: {DRAWS} draws each of 1 - 1/e of the clusters, without or with replacement")
    out()
    out(f"{'dataset':>8} {'arm':>12} {'07 full':>8} {'07 draws':>9} {'shrink':>7} {'without':>8} {'with':>8} {'copies':>7}")
    passed = True
    for name in NAMES:
        full = next(r["costs"] for r in seven if r["name"] == name and r["draw"] == 0)
        draws7 = [r["costs"] for r in seven if r["name"] == name and r["draw"] > 0]
        for c in RULE:
            f = gap(full, c)
            shrink = np.mean([gap(d, c) for d in draws7]) - f
            without = np.mean([gap(r["costs"], c) for r in done if r["name"] == name and not r["replace"]])
            with_ = np.mean([gap(r["costs"], c) for r in done if r["name"] == name and r["replace"]])
            out(f"{name:>8} {c:>12} {f:>+8.2f} {f + shrink:>+9.2f} {shrink:>+7.2f} {without:>+8.2f} {with_:>+8.2f} {with_ - without:>+7.2f}")
            if name == "meld":
                passed &= with_ - without >= shrink / 2
    out()
    out(f"copies explain at least half of the 07 shrink for every arm on meld: {passed}")
    save()


if __name__ == "__main__":
    main()
