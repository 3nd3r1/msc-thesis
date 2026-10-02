import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade import cheapest, indices, majority, sweep
from embed import cluster
from metrics import score
from report import report


RESULTS = Path(__file__).resolve().parent / "results"
LABELS = Path(__file__).resolve().parents[2] / "labels" / "all_beauty"

SEED = 0
REPEATS = 20
KS = [2, 3, 5, 8, 12, 16]
THRESHOLDS = [0.5, 0.6, 0.8, 1.0]
TARGETS = [0.90, 0.95, 0.99]

CONFIGS = [(0, 0.0)] + [(k, t) for k in KS for t in THRESHOLDS]
PREDICATES = [
    "reports skin irritation or an allergic reaction",
    "says the product doesn't work as advertised",
    "suspects the product is fake",
    "mentions having sensitive skin",
    "bought it as a gift",
    "mentions another person (partner, child, friend)",
    "the review is positive",
]


def embedding(rows):
    text = (rows.title.fillna("") + ". " + rows.text.fillna("")).tolist()
    return cluster(text, rows.parent_asin.nunique(), SEED)


GROUPINGS = {
    "product": lambda rows: rows.parent_asin,
    "rating": lambda rows: rows.rating,
    "product+rating": lambda rows: rows.parent_asin + "/" + rows.rating.astype(str),
    "embedding": embedding,
}


def verdicts(rows):
    labels = pd.read_json(LABELS / "labels.jsonl", lines=True)
    wide = labels.pivot(index="row", columns="predicate", values="verdict")
    wide = wide.reindex(rows.index)
    return {p: (wide[p] == "yes").to_numpy() for p in PREDICATES if p in wide}


def config(best):
    if best["k"] is None:
        return "full oracle"
    if best["k"] == 0:
        return "majority"
    return f"k={best['k']}, thr {best['threshold']:.1f}"


def replay():
    rows = pd.read_json(LABELS / "sample.jsonl", lines=True)
    truths = verdicts(rows)
    rng = np.random.default_rng(SEED)

    out, save = report(RESULTS, "replay")
    out(f"{len(rows):,} rows, {len(truths)} predicates, seed {SEED}")
    out(f"calls is a fraction of all rows, mean over {REPEATS} draws")
    out("k is rows sampled per group, thr the sample agreement needed to propagate")
    out("majority answers the global majority everywhere and is charged no calls")
    out("base is that same majority guess, for comparison")

    for name, keys in GROUPINGS.items():
        groups = indices(keys(rows))
        out(f"\n{name}: {len(groups):,} groups")

        for predicate, truth in truths.items():
            results = sweep(groups, truth, CONFIGS, REPEATS, rng)
            base, base_f1 = score(truth, majority(truth))
            out(f"\n  {predicate}, base {base:.1%} accuracy and {base_f1:.2f} f1")
            out(f"    {'target':>6}  {'calls':>6} {'acc':>6} {'f1':>5}  config")
            for target in TARGETS:
                best = cheapest(results, target)
                out(
                    f"    {target:>6.0%}  {best['calls']:>6.1%}"
                    f" {best['accuracy']:>6.1%} {best['f1']:>5.2f}  {config(best)}"
                )
    save()


STEPS = {"replay": replay}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"usage: run.py [{'|'.join(STEPS)}]")
    STEPS[step]()
