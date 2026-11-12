"""The CAD screen on another dataset: do its graphs carry label information the embedding does not?"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT / "experiments/03_link_guided_search"))
sys.path.insert(0, str(ROOT / "playground/synergy_screen"))
sys.path.insert(0, str(ROOT / "playground/cad_screen"))

from loaders import DATASETS
from embed import embed
from links import share
from run import correlation
from screen import ctrl, knn, lift_of
from search import adjacency, unit

SEEDS = 3


def ap(features, mask, seed):
    score = np.zeros(len(mask))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(features, mask):
        fit = LogisticRegression(max_iter=2000).fit(features[train], mask[train])
        score[test] = fit.decision_function(features[test])
    return average_precision_score(mask, score)


name = sys.argv[1]
rows, graphs = DATASETS[name]()
n = len(rows)
vectors = unit(embed([t or "(no text)" for t in rows.text]))
masks = {p: rows[p].to_numpy() for p in rows.columns if p != "text"}

print(f"{name}, {n:,} rows")
print("lift: edges leaving a positive that land on one, over the rate, ctrl is shuffled labels")
print("far: lift on the graph edges that are not in the matched cosine kNN, knn: lift on the kNN")
print(f"+knn: out of fold AP of emb + share of positive kNN neighbours, mean of {SEEDS} fold seeds")
print("gain: AP added by the share of positive graph neighbours, shuffled keeps every row's degree")
print("neighbour labels are true labels, so the gains are an upper bound")
print()
print(f"{'graph':>12} {'edges':>9} {'reach':>6} {'jacc':>6} {'predicate':>16} {'rate':>6} "
      f"{'lift':>6} {'ctrl':>6} {'far':>6} {'knn':>6} {'+knn':>6} {'gain':>14} {'shuffled':>14}")
for g, e in graphs.items():
    adj = adjacency(e, n)
    fake = adjacency(np.random.default_rng(0).permutation(n)[e], n)
    cos = knn(vectors, max(1, round(len(e) / n)))
    cadj = adjacency(cos, n)
    es, cs = set(map(tuple, e)), set(map(tuple, cos))
    far = np.array(sorted(es - cs))
    jacc = len(es & cs) / len(es | cs)
    reach = np.mean([len(x) > 0 for x in adj])
    for p, m in masks.items():
        base = np.c_[vectors, share(m, cadj)]
        runs = np.array([
            [ap(base, m, s), ap(np.c_[base, share(m, adj)], m, s), ap(np.c_[base, share(m, fake)], m, s)]
            for s in range(SEEDS)
        ])
        real, shuf = runs[:, 1] - runs[:, 0], runs[:, 2] - runs[:, 0]
        print(f"{g:>12} {len(e):>9,} {reach:>6.1%} {jacc:>6.3f} {p:>16} {m.mean():>6.1%} "
              f"{correlation(m, adj)['lift']:>5.1f}x {ctrl(m, adj):>5.1f}x {lift_of(far, m):>5.1f}x "
              f"{correlation(m, cadj)['lift']:>5.1f}x {runs[:, 0].mean():>6.3f} "
              f"{real.mean():>+7.3f}+-{real.std():.3f} {shuf.mean():>+7.3f}+-{shuf.std():.3f}",
              flush=True)
    print(flush=True)
