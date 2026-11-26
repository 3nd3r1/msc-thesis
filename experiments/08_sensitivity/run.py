"""The 07 arms under QBCB at other k and recall targets, on the same CAL orders."""

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
from guarantee import bargain, cal, clusters, knn_draw, qbcb, remap, resample, rewire, stop_count
from report import report
from search import adjacency

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["cad", "meld", "cora", "stack"]
TESTED = ["cad", "meld"]
DELTA = 0.1
SETTINGS = [(40, 0.9), (22, 0.9), (80, 0.9), (40, 0.8), (80, 0.95)]
CHECK, RULE_SETTINGS = SETTINGS[0], SETTINGS[1:]
TARGETS = sorted({t for _, t in SETTINGS})
GRID = [400, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
GRID_RUNS = 50
DRAWS = 200
FULL_RUNS, DRAW_RUNS = 2000, 1000
FULL_BARGAIN, DRAW_BARGAIN = 50, 20
ARMS = ["cal", "cal+knn", "cal+rewired", "cal+links"]
RULE = ["cal", "cal+knn", "cal+rewired"]
OTHERS = RULE + ["bargain"]
FLOOR = 0.88
WORKERS = int(os.environ.get("WORKERS", max(1, (os.cpu_count() or 2) // 2)))
BUDGETS = RESULTS / "budgets.jsonl"
DONE = RESULTS / "draws.jsonl"
DATA = {}


def label(k, t):
    return f"k{k} t{t}"


def size(b, n):
    return int(b) if b >= 1 else max(1, round(b * n))


def best_budget(job):
    """The budget on the grid with the lowest BARGAIN plus checking cost on the full data, picked in hindsight."""
    name, key, t = job
    y, p = DATA[name]["predicates"][key]
    cost = {b: np.mean([bargain(p, y, size(b, len(y)), t, DELTA, s)[0] for s in range(GRID_RUNS)]) for b in GRID}
    print(name, key, t, flush=True)
    return name, key, t, min(GRID, key=cost.get)


def costs(name, rows, copy, seed, runs, bargain_runs):
    """Cost and share of runs at the target of every arm per setting and predicate on one set of rows, cost in points of the rows."""
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
    out = {label(k, t): {} for k, t in SETTINGS}
    for key, (y, p) in d["predicates"].items():
        y, p = y[rows], p[rows]
        orders = {arm: cal(p, y, X, nb) for arm, nb in graphs.items()}
        bar = {}
        for t in TARGETS:
            budget = size(d["budgets"][key, t], n)
            res = np.array([bargain(p, y, budget, t, DELTA, seed * 100 + s) for s in range(bargain_runs)])
            bar[t] = 100 * res[:, 0].mean(), res[:, 1].mean()
        for k, t in SETTINGS:
            r = stop_count(k, t, DELTA)
            row = {}
            for arm, o in orders.items():
                rng = np.random.default_rng(seed)
                res = np.array([qbcb(o, y, k, r, t, rng) for _ in range(runs)])
                row[arm], row[arm + " met"] = 100 * res[:, 0].mean(), res[:, 1].mean()
            row["bargain"], row["bargain met"] = bar[t]
            out[label(k, t)][key] = row
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


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def load():
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
            "budgets": {},
            "predicates": {key: (masks[key][mask], np.nan_to_num(scores[key][mask]) + tie) for key, _, _ in ds.predicates},
        }


def run(jobs, fn, path, line):
    """Runs the jobs in a fresh pool, so the workers fork with DATA as it is now, and appends every result to path."""
    if not jobs:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("fork")) as pool, path.open("a") as f:
        for fut in as_completed([pool.submit(fn, j) for j in jobs]):
            f.write(json.dumps(line(*fut.result())) + "\n")
            f.flush()


def main():
    load()

    have = {(b["name"], b["predicate"], b["target"]) for b in read(BUDGETS)}
    jobs = [(name, key, t) for name in NAMES for key in DATA[name]["predicates"] for t in TARGETS]
    run([j for j in jobs if j not in have], best_budget, BUDGETS,
        lambda name, key, t, b: {"name": name, "predicate": key, "target": t, "budget": b})
    for b in read(BUDGETS):
        DATA[b["name"]]["budgets"][b["predicate"], b["target"]] = b["budget"]

    seen = {(r["name"], r["draw"]) for r in read(DONE)}
    jobs = [(name, 0) for name in NAMES] + [(name, d) for d in range(1, DRAWS + 1) for name in TESTED]
    run([j for j in jobs if j not in seen], one, DONE, lambda name, draw, res: {"name": name, "draw": draw, "costs": res})

    done = read(DONE)
    full = {r["name"]: r["costs"] for r in done if r["draw"] == 0}
    draws = {name: [r["costs"] for r in done if r["name"] == name and 1 <= r["draw"] <= DRAWS] for name in TESTED}

    cols = ["bargain"] + ARMS
    out, save = report(RESULTS, "costs")
    out(f"delta {DELTA}, pool with junk and repeated texts dropped")
    out("cost: distinct oracle calls as a share of the rows, every returned row confirmed, asking every row costs 100%")
    out("cal arms: QBCB with k and r as given over CAL, the 8B p(yes) order until both classes are seen, then a logistic")
    out("regression on the proxy logit and the embedding refit on the verdicts every 1% of the rows, plus counts of confirmed")
    out("positive and negative neighbours over no graph, a cosine kNN with k matched to the mean link degree, the links")
    out(f"rewired with every degree kept, or the links, mean of {FULL_RUNS} runs on the full data, the same orders at every setting")
    out(f"bargain: BARGAIN_R on the 8B p(yes) then checking, at the budget in {GRID} with the lowest cost")
    out(f"on the full data at each target, picked in hindsight, from results/budgets.jsonl, mean of {FULL_BARGAIN} runs")
    out("met: lowest share of runs reaching the target over the cal arms")
    out(f"k {CHECK[0]} target {CHECK[1]} is the 07 setting, a check that the full data reproduces 07, not part of the rule")
    for k, t in SETTINGS:
        s = label(k, t)
        out()
        out(f"k {k}, r {stop_count(k, t, DELTA)}, recall target {t}")
        out()
        out(f"{'dataset':>8} {'predicate':>13} {'rate':>6} " + " ".join(f"{c:>11}" for c in cols) + f" {'met':>5}")
        for name in NAMES:
            for key, r in full[name][s].items():
                rate = DATA[name]["predicates"][key][0].mean()
                met = min(r[a + " met"] for a in ARMS)
                out(f"{name:>8} {key:>13} {rate:>6.1%} " + " ".join(f"{r[c]:>10.1f}%" for c in cols) + f" {met:>5.0%}")
    out()

    out("cal+links minus each arm, mean over predicates, in points of the rows, full data then the 95% interval")
    out(f"over {DRAWS} cluster bootstrap draws of the link components, threads on CAD and dialogues on MELD, each draw")
    out(f"rebuilds the kNN among the drawn rows, redoes the rewiring and refits every order, {DRAW_RUNS} and {DRAW_BARGAIN} runs")
    out()
    out(f"{'k':>3} {'target':>6} {'dataset':>8} " + " ".join(f"{c:>22}" for c in OTHERS))
    upper = {}
    for k, t in SETTINGS:
        s = label(k, t)
        for name in NAMES:
            cells = []
            for c in OTHERS:
                cell = f"{gap(full[name][s], c):+.2f}"
                if name in TESTED:
                    lo, hi = np.percentile([gap(d[s], c) for d in draws[name]], [2.5, 97.5])
                    upper[s, name, c] = hi
                    cell += f" [{lo:+.2f}, {hi:+.2f}]"
                cells.append(cell)
            out(f"{k:>3} {t:>6} {name:>8} " + " ".join(f"{c:>22}" for c in cells))
    out()

    valid = all(r[a + " met"] >= FLOOR for name in NAMES for k, t in RULE_SETTINGS for r in full[name][label(k, t)].values() for a in ARMS)
    out(f"every cal cell at every tested setting meets the target in at least {FLOOR:.0%} of runs: {valid}")
    verdicts = []
    for k, t in RULE_SETTINGS:
        ok = all(upper[label(k, t), name, c] < 0 for name in TESTED for c in RULE)
        verdicts.append(ok)
        out(f"k {k}, target {t}: cal+links has the whole interval below 0 against {', '.join(RULE)} on {' and '.join(TESTED)}: {ok}")
    out(f"verdict: {'invalid' if not valid else 'pass' if all(verdicts) else 'fail'}")
    save()


if __name__ == "__main__":
    main()
