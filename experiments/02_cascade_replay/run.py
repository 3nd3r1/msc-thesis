import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade import calls_at_recall, indices, run
from embed import embed, kmeans
from metrics import score
from report import report


RESULTS = Path(__file__).resolve().parent / "results"
LABELS = Path(__file__).resolve().parents[2] / "labels" / "all_beauty"

SEED = 0
REPEATS = 10
CSV_CLUSTERS = 4
FRACTION = 0.005
FLOOR = 1
TARGETS = [0.90, 0.95, 0.99]
RECALL = 0.90

PREDICATES = {
    "reports skin irritation or an allergic reaction": "irritation",
    "says the product doesn't work as advertised": "doesn't work",
    "suspects the product is fake": "fake",
    "mentions having sensitive skin": "sensitive skin",
    "bought it as a gift": "gift",
    "mentions another person (partner, child, friend)": "another person",
    "the review is positive": "positive",
}


def variants(rows, vectors):
    return {
        "csv": kmeans(vectors, CSV_CLUSTERS, SEED),
        "rating": rows.rating,
        "product": rows.parent_asin,
        "product+rating": rows.parent_asin + "/" + rows.rating.astype(str),
    }


def verdicts(rows):
    labels = pd.read_json(LABELS / "labels.jsonl", lines=True)
    wide = labels.pivot(index="row", columns="predicate", values="verdict")
    wide = wide.reindex(rows.index)
    return {p: (wide[p] == "yes").to_numpy() for p in PREDICATES if p in wide}


def measure(groups, truth, vectors, target, rng):
    calls, accuracy, f1, recall_calls = [], [], [], []
    for _ in range(REPEATS):
        c, guess, trace = run(groups, truth, vectors, target, rng, SEED, FRACTION, FLOOR)
        a, f = score(truth, guess)
        calls.append(c)
        accuracy.append(a)
        f1.append(f)
        recall_calls.append(calls_at_recall(trace, truth.sum(), RECALL))
    reached = [c for c in recall_calls if c is not None]
    return {
        "calls": np.mean(calls),
        "calls_sd": np.std(calls),
        "accuracy": np.mean(accuracy),
        "f1": np.mean(f1),
        "hit": np.mean([a >= target for a in accuracy]),
        "recall_calls": np.mean(reached) if len(reached) == REPEATS else None,
    }


def table(out, header, cells, names, fmt, width=21):
    out(f"\n  {header}")
    out("    " + f"{'':>14}" + "".join(f"{n:>{width}}" for n in names))
    for short, row in cells.items():
        out("    " + f"{short:>14}" + "".join(fmt(row[n]) for n in names))


def replay():
    rows = pd.read_json(LABELS / "sample.jsonl", lines=True)
    truths = verdicts(rows)
    vectors = embed((rows.title.fillna("") + ". " + rows.text.fillna("")).tolist())
    groups = {n: indices(k) for n, k in variants(rows, vectors).items()}
    names = list(groups)
    rng = np.random.default_rng(SEED)

    out, save = report(RESULTS, "replay")
    out(f"{len(rows):,} rows, {len(truths)} predicates, seed {SEED}")
    out(f"{CSV_CLUSTERS} initial k-means clusters for csv")
    out(f"sample ceil({FRACTION} * group size) rows per group, at least {FLOOR}")
    out("propagate when the sample agrees at the target rate, otherwise split the rest")
    out("in two with k-means, and oracle any group at or below its sample size")
    out("\ninitial groups per variant: " + ", ".join(f"{n} {len(groups[n]):,}" for n in names))

    results = {t: {} for t in TARGETS}
    for target in TARGETS:
        for predicate, short in PREDICATES.items():
            truth = truths[predicate]
            results[target][short] = {
                n: measure(groups[n], truth, vectors, target, rng) for n in names
            }

    for target in TARGETS:
        cells = results[target]
        table(
            out,
            f"accuracy target {target:.0%}, calls% +- sd, f1, share of draws reaching it",
            cells,
            names,
            lambda r: (
                f"{100 * r['calls'] / len(rows):>7.1f}+-{100 * r['calls_sd'] / len(rows):<4.1f}"
                f" {r['f1']:>4.2f} {r['hit']:>4.0%}"
            ),
        )
        missed = [
            (short, n, r["accuracy"])
            for short, row in cells.items()
            for n, r in row.items()
            if r["accuracy"] < target
        ]
        if missed:
            out(f"    target missed: " + ", ".join(f"{n} on {s} at {a:.1%}" for s, n, a in missed))

    tightest = results[TARGETS[-1]]
    table(
        out,
        f"calls to find {RECALL:.0%} of the yes rows, on the {TARGETS[-1]:.0%} run",
        tightest,
        names,
        lambda r: (
            f"{r['recall_calls'] / len(rows):>21.1%}"
            if r["recall_calls"] is not None
            else f"{'never':>21}"
        ),
    )

    out("\n  cheapest variant that reaches the target, and its saving over csv")
    out("  none means no variant reached the target on every draw")
    out("    " + f"{'':>14}" + "".join(f"{f'@{t:.0%}':>24}" for t in TARGETS))
    for short in PREDICATES.values():
        line = f"    {short:>14}"
        for target in TARGETS:
            row = results[target][short]
            reaching = [n for n in names if row[n]["hit"] == 1]
            if not reaching:
                line += f"{'none':>24}"
                continue
            best = min(reaching, key=lambda n: row[n]["calls"])
            saving = 1 - row[best]["calls"] / row["csv"]["calls"]
            line += f"{best + f' {saving:+.0%}':>24}"
        out(line)
    save()


STEPS = {"replay": replay}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"usage: run.py [{'|'.join(STEPS)}]")
    STEPS[step]()
