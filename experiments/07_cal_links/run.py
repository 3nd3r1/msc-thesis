"""Verified positives at a guaranteed recall: QBCB over CAL with no, kNN, rewired or link neighbour features, and BARGAIN plus checking."""

import json
import os
import re
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
from guarantee import bargain, cal, clusters, knn_draw, qbcb, remap, resample, rewire, stop_count
from report import report
from search import adjacency

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["cad", "meld", "cora", "stack"]
TESTED = ["cad", "meld"]
T, DELTA, K = 0.9, 0.1, 40
R = stop_count(K, T, DELTA)
DRAWS = 200
FULL_RUNS, DRAW_RUNS = 2000, 1000
FULL_BARGAIN, DRAW_BARGAIN = 50, 20
ARMS = ["cal", "cal+knn", "cal+rewired", "cal+links"]
RULE = ["cal", "cal+knn", "cal+rewired"]
OTHERS = RULE + ["bargain"]
FLOOR = 0.88
WORKERS = max(1, (os.cpu_count() or 2) // 2)
DONE = RESULTS / "draws.jsonl"
DATA = {}


def budgets():
    """The budget of the wide BARGAIN best per predicate, as a share of the rows or a count."""
    out = {}
    for line in (ROOT / "playground/bargain_screen/results/wide.txt").read_text().splitlines():
        m = re.match(r"\s*(cad|meld|cora|stack)\s+(.+?)\s+[\d.]+%\s+[\d.]+%\s+(\S+)\s+\d+%", line)
        if m:
            out[m.group(1), m.group(2)] = float(m.group(3))
    return out


def costs(name, rows, copy, seed, runs, bargain_runs):
    """Cost and share of runs at the target of every arm per predicate on one set of rows, cost in points of the rows."""
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
            res = np.array([qbcb(o, y, K, R, T, rng) for _ in range(runs)])
            row[arm], row[arm + " met"] = 100 * res[:, 0].mean(), res[:, 1].mean()
        b = d["budgets"][key]
        budget = int(b) if b >= 1 else max(1, round(b * n))
        res = np.array([bargain(p, y, budget, T, DELTA, seed * 100 + s) for s in range(bargain_runs)])
        row["bargain"], row["bargain met"] = 100 * res[:, 0].mean(), res[:, 1].mean()
        out[key] = row
    return out


def one(job):
    """Draw 0 is the full data, every other draw a cluster bootstrap draw."""
    name, draw = job
    n = len(DATA[name]["group"])
    if draw == 0:
        res = costs(name, np.arange(n), np.zeros(n, int), 0, FULL_RUNS, FULL_BARGAIN)
    else:
        rows, copy = resample(DATA[name]["group"], np.random.default_rng(draw))
        res = costs(name, rows, copy, draw, DRAW_RUNS, DRAW_BARGAIN)
    print(name, draw, flush=True)
    return name, draw, res


def gap(res, c):
    return float(np.mean([r["cal+links"] - r[c] for r in res.values()]))


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
            "vectors": vectors_for(nodes)[mask],
            "k": max(1, round(len(links) / n)),
            "group": clusters(links, n),
            "budgets": {key: b[name, key] for key, _, _ in ds.predicates},
            "predicates": {key: (masks[key][mask], np.nan_to_num(scores[key][mask]) + tie) for key, _, _ in ds.predicates},
        }

    done = [json.loads(line) for line in DONE.read_text().splitlines()] if DONE.exists() else []
    seen = {(r["name"], r["draw"]) for r in done}
    jobs = [(name, 0) for name in NAMES] + [(name, d) for d in range(1, DRAWS + 1) for name in TESTED]
    jobs = [j for j in jobs if j not in seen]
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("fork")) as pool, DONE.open("a") as f:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            name, draw, res = fut.result()
            f.write(json.dumps({"name": name, "draw": draw, "costs": res}) + "\n")
            f.flush()
    done = [json.loads(line) for line in DONE.read_text().splitlines()]
    full = {r["name"]: r["costs"] for r in done if r["draw"] == 0}
    draws = {name: [r["costs"] for r in done if r["name"] == name and 1 <= r["draw"] <= DRAWS] for name in TESTED}

    cols = ["bargain"] + ARMS
    out, save = report(RESULTS, "costs")
    out(f"recall target {T}, delta {DELTA}, pool with junk and repeated texts dropped")
    out("cost: distinct oracle calls as a share of the rows, every returned row confirmed, asking every row costs 100%")
    out(f"cal arms: QBCB with k {K} and r {R} over CAL, the 8B p(yes) order until both classes are seen, then a logistic")
    out("regression on the proxy logit and the embedding refit on the verdicts every 1% of the rows, plus counts of confirmed")
    out("positive and negative neighbours over no graph, a cosine kNN with k matched to the mean link degree, the links")
    out(f"rewired with every degree kept, or the links, mean of {FULL_RUNS} runs on the full data")
    out("bargain: BARGAIN_R on the 8B p(yes) then checking, at the budget of its wide best from")
    out(f"playground/bargain_screen/results/wide.txt, mean of {FULL_BARGAIN} runs")
    out("met: lowest share of runs reaching the target over the cal arms")
    out()
    out(f"{'dataset':>8} {'predicate':>13} {'rate':>6} " + " ".join(f"{c:>11}" for c in cols) + f" {'met':>5}")
    for name in NAMES:
        for key, r in full[name].items():
            rate = DATA[name]["predicates"][key][0].mean()
            met = min(r[a + " met"] for a in ARMS)
            out(f"{name:>8} {key:>13} {rate:>6.1%} " + " ".join(f"{r[c]:>10.1f}%" for c in cols) + f" {met:>5.0%}")
    out()

    out(f"cal+links minus each arm, mean over predicates, in points of the rows, full data then the 95% interval")
    out(f"over {DRAWS} cluster bootstrap draws of the link components, threads on CAD and dialogues on MELD, each draw")
    out(f"rebuilds the kNN among the drawn rows, redoes the rewiring and refits every order, {DRAW_RUNS} and {DRAW_BARGAIN} runs")
    out()
    out(f"{'dataset':>8} " + " ".join(f"{c:>22}" for c in OTHERS))
    upper = {}
    for name in NAMES:
        cells = []
        for c in OTHERS:
            cell = f"{gap(full[name], c):+.2f}"
            if name in TESTED:
                g = [gap(d, c) for d in draws[name]]
                lo, hi = np.percentile(g, [2.5, 97.5])
                upper[name, c] = hi
                cell += f" [{lo:+.2f}, {hi:+.2f}]"
            cells.append(cell)
        out(f"{name:>8} " + " ".join(f"{c:>22}" for c in cells))
    out()

    valid = all(r[a + " met"] >= FLOOR for name in NAMES for r in full[name].values() for a in ARMS)
    passed = all(upper[name, c] < 0 for name in TESTED for c in RULE)
    out(f"every cal cell meets the target in at least {FLOOR:.0%} of runs: {valid}")
    out(f"cal+links has the whole interval below 0 against {', '.join(RULE)} on {' and '.join(TESTED)}: {passed}")
    out(f"verdict: {'invalid' if not valid else 'pass' if passed else 'fail'}")
    save()


if __name__ == "__main__":
    main()
