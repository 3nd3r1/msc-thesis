"""Do the proxy's errors correlate along the edges, and does that track what links add to it."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT / "experiments/03_link_guided_search"))

from report import report
from run import DATASETS, loaded, proxy_scores, subgraph
from search import adjacency, search

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["cora", "stack", "stack_author", "cad", "cad_thread", "cad_author", "meld", "meld_dialogue"]
TARGET = 0.90


def along(x, edges):
    """Pearson correlation of x between the two ends of every edge, both directions."""
    a = np.concatenate([edges[:, 0], edges[:, 1]])
    b = np.concatenate([edges[:, 1], edges[:, 0]])
    return np.corrcoef(x[a], x[b])[0, 1]


rows = []
for name in NAMES:
    ds = DATASETS[name]
    nodes, edges, _, masks, pools = loaded(ds)
    scores = proxy_scores(ds, len(nodes))
    mask = pools[1][1]
    sub = subgraph(mask, edges)
    neighbours = adjacency(sub, int(mask.sum()))
    for key, _, _ in ds.predicates:
        y = masks[key][mask].astype(float)
        p = scores[key][mask]
        calls = {o: search(y, o, None, neighbours, 0, (TARGET,), p)[0] for o in ("proxy", "proxy+links")}
        rows.append({
            "dataset": name,
            "predicate": key,
            "rate": y.mean(),
            "label_r": along(y, sub),
            "residual_r": along(y - p, sub),
            "gain": 100 * (calls["proxy+links"] - calls["proxy"]) / len(y),
        })
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
out, save = report(RESULTS, "residual")
out("label r: Pearson correlation of the label between the two ends of an edge")
out("residual r: the same for the label minus the proxy's p(yes)")
out(f"gain: proxy+links minus proxy in points of the rows at {TARGET:.0%} recall, negative means links help")
out("pool with junk and repeated texts dropped")
out()
out(f"{'dataset':>13} {'predicate':>13} {'rate':>6} {'label r':>8} {'residual r':>11} {'gain':>6}")
for r in df.itertuples():
    out(f"{r.dataset:>13} {r.predicate:>13} {r.rate:>6.1%} {r.label_r:>8.3f} {r.residual_r:>11.3f} {r.gain:>+6.1f}")
out()
out("Spearman correlation with gain over all predicates, negative means more correlation goes with more help")
for col in ("label_r", "residual_r"):
    out(f"  {col.replace('_', ' '):>10} {df[col].corr(df.gain, method='spearman'):+.3f}")
save()
