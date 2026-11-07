"""Do CAD's reply, thread and author graphs carry label information the embedding does not?"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT / "experiments/03_link_guided_search"))
sys.path.insert(0, str(ROOT / "playground/synergy_screen"))

from embed import embed
from run import correlation
from screen import ctrl, knn, lift_of, pairs_from
from search import adjacency, unit

CAD = ROOT / "data/cad/raw/data/cad_v1_1.tsv"
PREDICATES = [
    "IdentityDirectedAbuse",
    "AffiliationDirectedAbuse",
    "PersonDirectedAbuse",
    "CounterSpeech",
    "Slur",
]


def load():
    d = pd.read_csv(CAD, sep="\t", dtype=str, keep_default_na=False)
    d = d[~d.split.str.startswith("exclude")]
    rows = d.groupby("info_id", sort=True).agg(
        parent=("info_id.parent", "first"),
        thread=("info_thread.id", "first"),
        author=("meta_author", "first"),
        text=("meta_text", "first"),
        labels=("annotation_Primary", set),
        context=("annotation_Context", set),
    )
    return rows.reset_index()


def graphs(rows):
    index = {k: i for i, k in enumerate(rows.info_id)}
    reply = {
        (min(i, index[p]), max(i, index[p]))
        for i, p in enumerate(rows.parent)
        if p in index and index[p] != i
    }
    groups = lambda col, skip=(): [
        g for k, g in rows.groupby(col).groups.items() if k not in skip
    ]
    return {
        "reply": np.array(sorted(reply)),
        "thread": pairs_from(groups("thread"), cap=10**9),
        "author": pairs_from(groups("author", ("", "[deleted]")), cap=10**9),
    }


def share(mask, neighbours):
    """Share of positive neighbours and whether there are any, from the true labels."""
    deg = np.array([len(x) for x in neighbours])
    pos = np.array([mask[x].sum() for x in neighbours])
    return np.c_[pos / np.maximum(deg, 1), deg > 0]


def ap(features, mask):
    score = np.zeros(len(mask))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(features, mask):
        fit = LogisticRegression(max_iter=2000).fit(features[train], mask[train])
        score[test] = fit.decision_function(features[test])
    return average_precision_score(mask, score)


if __name__ == "__main__":
    rows = load()
    n = len(rows)
    vectors = unit(embed([t or "(no text)" for t in rows.text]))
    masks = {p: rows.labels.map(lambda s: p in s).to_numpy() for p in PREDICATES}

    print(f"CAD v1.1, {n:,} entries, excluded splits dropped")
    print("lift: edges leaving a positive that land on one, over the rate, ctrl is shuffled labels")
    print("far: lift on the graph edges that are not in the matched cosine kNN")
    print("ap: out of fold logistic regression, emb alone, + share of positive kNN neighbours,")
    print("    + share of positive graph neighbours, neighbour labels are true labels so an upper bound")
    print()
    for p, m in masks.items():
        prev = rows.context[m].map(lambda s: "PreviousContent" in s).mean()
        print(f"{p:>26} {m.sum():>5,} {m.mean():>6.1%}  {prev:>6.1%} need the previous post")
    print()

    base = {p: ap(vectors, m) for p, m in masks.items()}
    print(f"{'graph':>8} {'edges':>8} {'reach':>6} {'jacc':>6} {'predicate':>26} "
          f"{'lift':>6} {'ctrl':>6} {'far':>6} {'knn':>6} {'ap emb':>7} {'+knn':>7} {'+graph':>7}")
    for name, e in graphs(rows).items():
        adj = adjacency(e, n)
        k = max(1, round(len(e) / n))
        cos = knn(vectors, k)
        cadj = adjacency(cos, n)
        es, cs = set(map(tuple, e)), set(map(tuple, cos))
        far = np.array(sorted(es - cs))
        jacc = len(es & cs) / len(es | cs)
        reach = np.mean([len(x) > 0 for x in adj])
        for p, m in masks.items():
            c = correlation(m, adj)
            cl = correlation(m, cadj)
            with_knn = ap(np.c_[vectors, share(m, cadj)], m)
            with_graph = ap(np.c_[vectors, share(m, cadj), share(m, adj)], m)
            print(f"{name:>8} {len(e):>8,} {reach:>6.1%} {jacc:>6.3f} {p:>26} "
                  f"{c['lift']:>5.1f}x {ctrl(m, adj):>5.1f}x {lift_of(far, m):>5.1f}x "
                  f"{cl['lift']:>5.1f}x {base[p]:>7.3f} {with_knn:>7.3f} {with_graph:>7.3f}",
                  flush=True)
        print(flush=True)
