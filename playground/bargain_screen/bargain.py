"""BARGAIN's recall target query on our datasets, with the 8B as the proxy and our labels as the oracle, so no calls."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from BARGAIN import BARGAIN_R, Oracle, Proxy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))

from corpora import DATASETS, loaded, proxy_scores
from report import report

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["cora", "stack", "cad", "meld"]
TARGET = 0.9
DELTA = 0.1
BUDGET = 400
RUNS = 50
VARIANTS = {"U": dict(beta=0), "A": dict(beta=0.02, r=150)}


class Scores(Proxy):
    def __init__(self, p):
        super().__init__(verbose=False)
        self.p = p

    def get_preds_and_scores(self, indxs, data_records):
        return np.ones(len(indxs), bool), self.p[np.asarray(indxs, int)]


class Labels(Oracle):
    def __init__(self, y):
        super().__init__(verbose=False)
        self.y = y

    def oracle_func(self, input, proxy_output):
        return True, bool(self.y[int(input)])


def ideal(p, y):
    """Precision of the smallest proxy prefix with recall at the target, known in hindsight."""
    hits = np.cumsum(y[np.argsort(-p)])
    k = int(np.searchsorted(hits, TARGET * y.sum())) + 1
    return hits[k - 1] / k


rows = []
for name in NAMES:
    ds = DATASETS[name]
    nodes, _, _, masks, pools = loaded(ds)
    mask = pools[1][1]
    scores = proxy_scores(ds, len(nodes))
    for key, _, _ in ds.predicates:
        y = masks[key][mask]
        p = np.nan_to_num(scores[key][mask])
        records = list(range(len(y)))
        for variant, params in VARIANTS.items():
            for seed in range(RUNS):
                oracle = Labels(y)
                found = np.array(BARGAIN_R(Scores(p), oracle, DELTA, TARGET, BUDGET, seed=seed, **params).process(records), int)
                hit = y[found].sum()
                rows.append({
                    "dataset": name,
                    "predicate": key,
                    "variant": variant,
                    "rate": y.mean(),
                    "ideal": ideal(p, y),
                    "precision": hit / max(len(found), 1),
                    "recall": hit / y.sum(),
                    "returned": len(found) / len(y),
                    "calls": oracle.get_number_preds(),
                })
        print(name, key, flush=True)

df = pd.DataFrame(rows)
g = df.groupby(["dataset", "predicate", "variant"], sort=False)
summary = g.agg(
    rate=("rate", "first"),
    ideal=("ideal", "first"),
    precision=("precision", "mean"),
    recall=("recall", "mean"),
    met=("recall", lambda r: (r >= TARGET).mean()),
    returned=("returned", "mean"),
    calls=("calls", "mean"),
).reset_index()

out, save = report(RESULTS, "bargain")
out(f"BARGAIN_R, recall target {TARGET}, delta {DELTA}, budget {BUDGET} oracle calls, mean of {RUNS} runs")
out("U is uniform sampling with the full guarantee, A is adaptive sampling with beta 0.02 and r 150 as in the paper")
out("proxy is the 8B p(yes), oracle is our labels, pool with junk and repeated texts dropped")
out("ideal: precision of the smallest proxy prefix reaching the target, known in hindsight")
out("met: share of runs with recall at the target, returned: share of rows returned, calls: distinct oracle calls")
out()
out(f"{'dataset':>8} {'predicate':>13} {'v':>2} {'rate':>6} {'ideal':>6} {'prec':>6} {'recall':>7} {'met':>5} {'returned':>9} {'calls':>6}")
for r in summary.itertuples():
    out(f"{r.dataset:>8} {r.predicate:>13} {r.variant:>2} {r.rate:>6.1%} {r.ideal:>6.1%} {r.precision:>6.1%} {r.recall:>7.1%} {r.met:>5.0%} {r.returned:>9.1%} {r.calls:>6.0f}")
save()
