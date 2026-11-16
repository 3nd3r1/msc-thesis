"""Do links add more to the proxy on predicates that are harder to find from the text alone.

Fixed before looking:
- text difficulty is the out of fold average precision of a logistic regression on the embeddings, 5 folds, seed 0, as in the ceiling step of 03
- pool with junk and repeated texts dropped
- gain is proxy+links minus proxy at 90% recall, as in refusal_screen
- passes if the Spearman of AP against gain over every predicate and edge set is +0.5 or higher, high AP meaning easy from text and positive gain meaning links do not help
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))

from corpora import DATASETS, loaded, proxy_scores, subgraph, vectors_for
from report import report
from search import adjacency, search

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["cora", "stack", "stack_author", "cad", "cad_thread", "cad_author", "meld", "meld_dialogue"]
TARGET = 0.90
PASS = 0.5


def text_ap(v, y):
    score = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(v, y):
        score[test] = LogisticRegression(max_iter=1000).fit(v[train], y[train]).decision_function(v[test])
    return average_precision_score(y, score)


ap = {}
rows = []
for name in NAMES:
    ds = DATASETS[name]
    nodes, edges, _, masks, pools = loaded(ds)
    mask = pools[1][1]
    scores = proxy_scores(ds, len(nodes))
    neighbours = adjacency(subgraph(mask, edges), int(mask.sum()))
    if ds.labels not in ap:
        v = vectors_for(nodes)[mask]
        ap[ds.labels] = {k: text_ap(v, masks[k][mask]) for k, _, _ in ds.predicates}
    for key, _, _ in ds.predicates:
        y = masks[key][mask].astype(float)
        p = scores[key][mask]
        calls = {o: search(y, o, None, neighbours, 0, (TARGET,), p)[0] for o in ("proxy", "proxy+links")}
        rows.append({
            "dataset": name,
            "predicate": key,
            "rate": y.mean(),
            "ap": ap[ds.labels][key],
            "gain": 100 * (calls["proxy+links"] - calls["proxy"]) / len(y),
        })
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
rho = df.ap.corr(df.gain, method="spearman")
out, save = report(RESULTS, "shape")
out("ap: out of fold average precision of a logistic regression on the embeddings, 5 folds, seed 0")
out(f"gain: proxy+links minus proxy in points of the rows at {TARGET:.0%} recall, negative means links help")
out("pool with junk and repeated texts dropped")
out()
out(f"{'dataset':>13} {'predicate':>13} {'rate':>6} {'ap':>6} {'gain':>6}")
for r in df.itertuples():
    out(f"{r.dataset:>13} {r.predicate:>13} {r.rate:>6.1%} {r.ap:>6.3f} {r.gain:>+6.1f}")
out()
out(f"Spearman of ap against gain over all predicates {rho:+.3f}, fixed to pass at +{PASS} or higher: {'pass' if rho >= PASS else 'fail'}")
save()
