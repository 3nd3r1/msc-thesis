"""BARGAIN's recall target query on scores from a first phase of oracle calls along a graph, against plain BARGAIN at the same budget.

Fixed before the run: the budget grid, the phase 1 share, the score rule, k of the kNN and the rewiring.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from BARGAIN import BARGAIN_R, Oracle, Proxy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))

from corpora import DATASETS, loaded, proxy_scores, subgraph, vectors_for
from report import report
from search import adjacency, knn

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = sys.argv[1:] or ["cora", "stack", "cad", "cad_thread", "meld", "meld_dialogue"]
TARGET = 0.9
DELTA = 0.1
FIXED = 400
SHARES = [0.01, 0.02, 0.05, 0.10]
SEARCH = 0.25
RUNS = 50
ARMS = ["plain", "proxy", "knn", "rewired", "links"]


class Scores(Proxy):
    def __init__(self, s):
        super().__init__(verbose=False)
        self.s = s

    def get_preds_and_scores(self, indxs, data_records):
        return np.ones(len(indxs), bool), self.s[np.asarray(indxs, int)]


class Labels(Oracle):
    def __init__(self, y, asked):
        super().__init__(verbose=False)
        self.y = y
        self.asked = set(np.flatnonzero(asked).tolist())

    def oracle_func(self, input, proxy_output):
        self.asked.add(int(input))
        return True, bool(self.y[int(input)])


def rewire(edges, seed=0):
    """Every row keeps its degree and its edges go to random rows."""
    stubs = np.random.default_rng(seed).permutation(edges.ravel()).reshape(-1, 2)
    return stubs[stubs[:, 0] != stubs[:, 1]]


def phase1(p, y, neighbours, calls):
    """Oracle calls highest first, with any neighbour of a found positive above every untouched row."""
    asked = np.zeros(len(y), bool)
    front = np.zeros(len(y))
    for _ in range(calls):
        i = int(np.argmax(np.where(asked, -np.inf, p + front)))
        asked[i] = True
        if y[i] and neighbours is not None:
            front[neighbours[i]] = 1
    score = p + front
    score[asked & y] = 3
    score[asked & ~y] = -1
    return score, asked


def runs(job):
    """RUNS seeds of BARGAIN on frozen scores, cost is the oracle calls to verify the returned set."""
    meta, y, s, asked, budget = job
    out = []
    records = list(range(len(y)))
    for seed in range(RUNS):
        oracle = Labels(y, asked)
        found = np.array(BARGAIN_R(Scores(s), oracle, DELTA, TARGET, budget, seed=seed).process(records), int)
        hit = y[found].sum()
        out.append(meta | {
            "precision": hit / max(len(found), 1),
            "recall": hit / y.sum(),
            "returned": len(found) / len(y),
            "cost": len(oracle.asked | set(found.tolist())) / len(y),
        })
    return out


def jobs():
    for name in NAMES:
        ds = DATASETS[name]
        nodes, edges, _, masks, pools = loaded(ds)
        mask = pools[1][1]
        n = int(mask.sum())
        links = subgraph(mask, edges)
        graphs = {
            "proxy": None,
            "knn": adjacency(knn(vectors_for(nodes)[mask], max(1, round(len(links) / n))), n),
            "rewired": adjacency(rewire(links), n),
            "links": adjacency(links, n),
        }
        scores = proxy_scores(ds, len(nodes))
        budgets = [FIXED] + [max(1, round(share * n)) for share in SHARES]
        for key, _, _ in ds.predicates:
            y = masks[key][mask]
            p = np.nan_to_num(scores[key][mask])
            for budget in budgets:
                meta = {"dataset": name, "predicate": key, "budget": budget, "rate": y.mean()}
                yield meta | {"arm": "plain"}, y, p, np.zeros(n, bool), budget
                calls = round(SEARCH * budget)
                for graph, neighbours in graphs.items():
                    s, asked = phase1(p, y, neighbours, calls)
                    yield meta | {"arm": graph}, y, s, asked, budget - calls
            print(name, key, flush=True)


with ProcessPoolExecutor(mp_context=get_context("fork")) as pool:
    rows = [r for out in pool.map(runs, jobs()) for r in out]

df = pd.DataFrame(rows)
df["arm"] = pd.Categorical(df.arm, ARMS)
g = df.groupby(["dataset", "predicate", "budget", "arm"], sort=False, observed=True)
summary = g.agg(
    rate=("rate", "first"),
    precision=("precision", "mean"),
    met=("recall", lambda r: (r >= TARGET).mean()),
    returned=("returned", "mean"),
    cost=("cost", "mean"),
).reset_index()
best = summary.groupby(["dataset", "predicate", "arm"], sort=False, observed=True).cost.min().unstack("arm")

out, save = report(RESULTS, "links")
out(f"BARGAIN_R uniform, recall target {TARGET}, delta {DELTA}, mean of {RUNS} runs, pool with junk and repeated texts dropped")
out(f"budget: {FIXED} oracle calls as in the paper, and {', '.join(f'{s:.0%}' for s in SHARES)} of the rows")
out(f"plain: BARGAIN on the 8B p(yes) with the full budget")
out(f"proxy, knn, rewired, links: {SEARCH:.0%} of the budget on search first, then BARGAIN with the rest on the frozen scores")
out("search asks the highest unasked row, a neighbour of a found positive goes above every untouched row")
out("frozen scores: found positives, then neighbours of positives by p, then the rest by p, then found negatives")
out("knn is the cosine kNN with k matched to the mean link degree, rewired keeps every row's link degree with random ends")
out("met: share of runs with recall at the target, returned: share of rows returned")
out("cost: distinct oracle calls over both phases plus the returned rows not yet asked, as a share of the rows")
out()
out("lowest cost over the budget grid")
out()
out(f"{'dataset':>13} {'predicate':>13} " + " ".join(f"{a:>8}" for a in ARMS))
for (dataset, predicate), r in best.iterrows():
    out(f"{dataset:>13} {predicate:>13} " + " ".join(f"{r[a]:>8.1%}" for a in ARMS))
out()
out("every budget")
out()
out(f"{'dataset':>13} {'predicate':>13} {'budget':>6} {'arm':>7} {'rate':>6} {'prec':>6} {'met':>5} {'returned':>9} {'cost':>6}")
for r in summary.itertuples():
    out(f"{r.dataset:>13} {r.predicate:>13} {r.budget:>6} {r.arm:>7} {r.rate:>6.1%} {r.precision:>6.1%} {r.met:>5.0%} {r.returned:>9.1%} {r.cost:>6.1%}")
save()
