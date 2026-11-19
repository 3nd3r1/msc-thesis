"""CAL with and without link features as the order under QBCB, against wide BARGAIN and the 06 orders."""

import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))

from corpora import DATASETS, loaded, proxy_scores, subgraph, vectors_for
from guarantee import cal, qbcb, stop_count
from report import report
from search import adjacency

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = sys.argv[1:] or ["cad", "meld", "cora", "stack"]
T, DELTA, K = 0.9, 0.1, 40
R = stop_count(K, T, DELTA)
RUNS = 2000
ARMS = ["cal", "cal+links"]


def earlier():
    """wide BARGAIN, and QBCB over the proxy and link orders, per predicate."""
    out = {}
    for line in (ROOT / "playground/bargain_screen/results/wide.txt").read_text().splitlines():
        m = re.match(r"\s*(cad|meld|cora|stack)\s+(.+?)\s+([\d.]+)%\s+([\d.]+)%\s+\S+\s+\d+%\s+([\d.]+)%\s+([\d.]+)%", line)
        if m:
            out[m.group(1), m.group(2)] = {"bargain": float(m.group(4)), "proxy": float(m.group(5)), "links": float(m.group(6))}
    return out


def one(job):
    name, key, y, p, X, nb = job
    row = {"dataset": name, "predicate": key, "rate": y.mean()}
    for arm, graph in zip(ARMS, [None, nb]):
        o = cal(p, y, X, graph)
        rng = np.random.default_rng(0)
        res = np.array([qbcb(o, y, K, R, T, rng) for _ in range(RUNS)])
        row[arm] = 100 * res[:, 0].mean()
        row[arm + " met"] = res[:, 1].mean()
    print(name, key, flush=True)
    return row


def jobs():
    for name in NAMES:
        ds = DATASETS[name]
        nodes, edges, _, masks, pools = loaded(ds)
        mask = pools[1][1]
        n = int(mask.sum())
        nb = adjacency(subgraph(mask, edges), n)
        X = vectors_for(nodes)[mask]
        scores = proxy_scores(ds, len(nodes))
        tie = np.random.default_rng(0).random(n) * 1e-9
        for key, _, _ in ds.predicates:
            yield name, key, masks[key][mask], np.nan_to_num(scores[key][mask]) + tie, X, nb


with ProcessPoolExecutor(mp_context=get_context("fork")) as pool:
    rows = list(pool.map(one, jobs()))
prev = earlier()
cols = ["bargain", "proxy", "links"] + ARMS

out, save = report(RESULTS, "cal" if NAMES == ["cad", "meld", "cora", "stack"] else "_".join(NAMES))
out(f"recall target {T}, delta {DELTA}, QBCB with k {K} and r {R}, mean of {RUNS} runs")
out("cost: distinct oracle calls as a share of the rows, every returned row confirmed")
out("bargain, proxy, links: wide BARGAIN plus checking and QBCB over the 06 orders, from playground/bargain_screen/results/wide.txt")
out("cal: the 8B p(yes) order until both classes are seen, then a logistic regression on the proxy logit and the")
out("embedding refit on the verdicts every 1% of the rows, cal+links adds counts of confirmed positive and negative neighbours")
out("met: lowest share of runs at the target over the two cal arms")
out()
out(f"{'dataset':>8} {'predicate':>13} {'rate':>6} " + " ".join(f"{c:>9}" for c in cols) + f" {'met':>5}")
for r in rows:
    r |= prev[r["dataset"], r["predicate"]]
    met = min(r[a + " met"] for a in ARMS)
    out(f"{r['dataset']:>8} {r['predicate']:>13} {r['rate']:>6.1%} " + " ".join(f"{r[c]:>8.1f}%" for c in cols) + f" {met:>5.0%}")
out()
out("mean over predicates of each arm minus wide BARGAIN, in points of the rows")
out()
out(f"{'dataset':>8} " + " ".join(f"{c:>9}" for c in cols[1:]))
for name in NAMES:
    sub = [r for r in rows if r["dataset"] == name]
    out(f"{name:>8} " + " ".join(f"{np.mean([r[c] - r['bargain'] for r in sub]):>+9.2f}" for c in cols[1:]))
out()
out("cal+links minus cal, mean over predicates")
out()
for name in NAMES:
    sub = [r for r in rows if r["dataset"] == name]
    out(f"{name:>8} {np.mean([r['cal+links'] - r['cal'] for r in sub]):>+9.2f}")
save()
