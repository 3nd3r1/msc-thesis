"""BARGAIN plus checking with the budget grid of 06 widened to half the rows, against QBCB from 06."""

import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from BARGAIN import BARGAIN_R, Oracle, Proxy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))

from corpora import DATASETS, loaded, proxy_scores
from report import report

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["cad", "meld", "cora", "stack"]
T, DELTA = 0.9, 0.1
OLD = [400, 0.01, 0.02, 0.05, 0.10]
NEW = [0.20, 0.30, 0.50]
RUNS = 50


class Scores(Proxy):
    def __init__(self, s):
        super().__init__(verbose=False)
        self.s = s

    def get_preds_and_scores(self, indxs, data_records):
        return np.ones(len(indxs), bool), self.s[np.asarray(indxs, int)]


class Labels(Oracle):
    def __init__(self, y):
        super().__init__(verbose=False)
        self.y = y
        self.asked = set()

    def oracle_func(self, input, proxy_output):
        self.asked.add(int(input))
        return True, bool(self.y[int(input)])


def bargain(p, y, budget, seed):
    oracle = Labels(y)
    found = np.array(BARGAIN_R(Scores(p), oracle, DELTA, T, budget, seed=seed).process(list(range(len(y)))), int)
    return len(oracle.asked | set(found.tolist())) / len(y), y[found].sum() >= T * y.sum()


def one(job):
    name, key, y, p = job
    n = len(y)
    out = {}
    for b in OLD + NEW:
        budget = b if b >= 1 else max(1, round(b * n))
        res = np.array([bargain(p, y, budget, s) for s in range(RUNS)])
        out[b] = res.mean(0)
    print(name, key, flush=True)
    return name, key, out


def jobs():
    for name in NAMES:
        ds = DATASETS[name]
        nodes, _, _, masks, pools = loaded(ds)
        mask = pools[1][1]
        n = int(mask.sum())
        scores = proxy_scores(ds, len(nodes))
        tie = np.random.default_rng(0).random(n) * 1e-9
        for key, _, _ in ds.predicates:
            yield name, key, masks[key][mask], np.nan_to_num(scores[key][mask]) + tie


def qbcb():
    """proxy and links cost per predicate from 06."""
    out = {}
    for line in (ROOT / "experiments/06_tar_links/results/costs.txt").read_text().splitlines():
        m = re.match(r"\s*(cad|meld|cora|stack)\s+(.+?)\s+[\d.]+%\s+\d+\s+(.*)", line)
        if m:
            costs = [float(c) for c in re.findall(r"([\d.]+)%", m.group(3))]
            out[m.group(1), m.group(2)] = {"proxy": costs[2], "links": costs[5]}
    return out


with ProcessPoolExecutor(mp_context=get_context("fork")) as pool:
    rows = list(pool.map(one, jobs()))
q = qbcb()

out, save = report(RESULTS, "wide")
out(f"BARGAIN_R on the 8B p(yes) then checking, recall target {T}, delta {DELTA}, mean of {RUNS} runs")
out(f"old: lowest cost over the 06 budgets {OLD}, wide: also {NEW} of the rows, both picked in hindsight")
out("budget: the budget of the wide best, met: share of runs at the target for it")
out("proxy and links: QBCB from experiments/06_tar_links/results/costs.txt")
out()
out(f"{'dataset':>8} {'predicate':>13} {'old':>7} {'wide':>7} {'budget':>7} {'met':>5} {'proxy':>7} {'links':>7}")
diffs = {name: [] for name in NAMES}
for name, key, res in rows:
    best = min(OLD + NEW, key=lambda b: res[b][0])
    old = min(res[b][0] for b in OLD)
    wide = res[best][0]
    p, l = q[name, key]["proxy"], q[name, key]["links"]
    diffs[name].append((p - 100 * wide, l - 100 * wide))
    out(f"{name:>8} {key:>13} {old:>7.1%} {wide:>7.1%} {str(best):>7} {res[best][1]:>5.0%} {p:>6.1f}% {l:>6.1f}%")
out()
out("mean over predicates of QBCB minus wide BARGAIN, in points of the rows")
out()
out(f"{'dataset':>8} {'proxy':>7} {'links':>7}")
for name in NAMES:
    d = np.array(diffs[name])
    out(f"{name:>8} {d[:, 0].mean():>+7.2f} {d[:, 1].mean():>+7.2f}")
save()
