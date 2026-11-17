"""Do links add more to the proxy on predicates that are hard from the text and whose neighbours share the label.

Fixed before looking:
- ap and gain are read from results/shape.txt
- lift is the share of edges leaving a positive that land on one over the positive rate, as in the lift step of 03, on the same pool
- joint is (1 - ap) * (lift - 1), high meaning hard from the text and clustered along the edges
- passes if all three hold:
  - the Spearman of joint against gain over every predicate and edge set is -0.5 or lower
  - the same over the cad and meld edge sets alone is -0.3 or lower, so it cannot pass on the split between datasets alone
  - over every predicate the joint is further from 0 than ap and lift alone
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))

from corpora import DATASETS, correlation, loaded, subgraph
from report import report
from search import adjacency

RESULTS = Path(__file__).resolve().parent / "results"
ALL = -0.5
WITHIN = -0.3

df = pd.read_fwf(
    RESULTS / "shape.txt",
    skiprows=5,
    skipfooter=2,
    colspecs=[(0, 13), (14, 27), (28, 34), (35, 41), (42, 48)],
    names=["dataset", "predicate", "rate", "ap", "gain"],
    engine="python",
)

lift = {}
for name in df.dataset.unique():
    ds = DATASETS[name]
    nodes, edges, _, masks, pools = loaded(ds)
    mask = pools[1][1]
    neighbours = adjacency(subgraph(mask, edges), int(mask.sum()))
    for key, _, _ in ds.predicates:
        lift[name, key] = correlation(masks[key][mask], neighbours)["lift"]

df["lift"] = [lift[d, p] for d, p in zip(df.dataset, df.predicate)]
df["joint"] = (1 - df.ap) * (df.lift - 1)
inside = df[df.dataset.str.startswith(("cad", "meld"))]
rho = {c: df[c].corr(df.gain, method="spearman") for c in ("ap", "lift", "joint")}
rho_in = inside.joint.corr(inside.gain, method="spearman")
ok = rho["joint"] <= ALL and rho_in <= WITHIN and abs(rho["joint"]) > max(abs(rho["ap"]), abs(rho["lift"]))

out, save = report(RESULTS, "joint")
out("ap and gain from results/shape.txt")
out("lift: share of the edges leaving a positive that land on one, over the positive rate")
out("joint: (1 - ap) * (lift - 1)")
out("gain: proxy+links minus proxy in points of the rows at 90% recall, negative means links help")
out("pool with junk and repeated texts dropped")
out()
out(f"{'dataset':>13} {'predicate':>13} {'ap':>6} {'lift':>6} {'joint':>6} {'gain':>6}")
for r in df.itertuples():
    out(f"{r.dataset:>13} {r.predicate:>13} {r.ap:>6.3f} {r.lift:>5.1f}x {r.joint:>6.2f} {r.gain:>+6.1f}")
out()
out("Spearman against gain over all predicates")
for c, v in rho.items():
    out(f"  {c:>5} {v:+.3f}")
out(f"Spearman of joint against gain over cad and meld {rho_in:+.3f}")
out(f"fixed to pass at {ALL} or lower over all, {WITHIN} or lower over cad and meld, and joint further from 0 than ap and lift: {'pass' if ok else 'fail'}")
save()
