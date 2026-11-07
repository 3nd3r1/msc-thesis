"""Is the AP gain from the graphs in links.py larger than fold noise and a shuffled graph?"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from links import PREDICATES, graphs, knn, load, share
from embed import embed
from search import adjacency, unit

SEEDS = 5


def ap(features, mask, seed):
    score = np.zeros(len(mask))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(features, mask):
        fit = LogisticRegression(max_iter=2000).fit(features[train], mask[train])
        score[test] = fit.decision_function(features[test])
    return average_precision_score(mask, score)


rows = load()
n = len(rows)
vectors = unit(embed([t or "(no text)" for t in rows.text]))
masks = {p: rows.labels.map(lambda s: p in s).to_numpy() for p in PREDICATES}

print(f"AP gain over emb + kNN neighbours, mean and sd of {SEEDS} fold seeds")
print("shuffled keeps every row's degree and connects it to random rows")
print()
print(f"{'graph':>8} {'predicate':>26} {'+knn':>7} {'graph gain':>14} {'shuffled gain':>14}")
for name, e in graphs(rows).items():
    adj = adjacency(e, n)
    fake = adjacency(np.random.default_rng(0).permutation(n)[e], n)
    cadj = adjacency(knn(vectors, max(1, round(len(e) / n))), n)
    for p, m in masks.items():
        base = np.c_[vectors, share(m, cadj)]
        runs = np.array([
            [ap(base, m, s), ap(np.c_[base, share(m, adj)], m, s), ap(np.c_[base, share(m, fake)], m, s)]
            for s in range(SEEDS)
        ])
        real, shuf = runs[:, 1] - runs[:, 0], runs[:, 2] - runs[:, 0]
        print(f"{name:>8} {p:>26} {runs[:, 0].mean():>7.3f} "
              f"{real.mean():>+7.3f}+-{real.std():.3f} {shuf.mean():>+7.3f}+-{shuf.std():.3f}",
              flush=True)
    print(flush=True)
