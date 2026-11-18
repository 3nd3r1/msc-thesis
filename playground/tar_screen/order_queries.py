"""Positions of every positive in each adaptive order, then budgeted recall and a QBCB stop (Lewis et al. 2021)."""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from scipy.stats import binom

sys.path.insert(0, "/home/ender/uni/msc-thesis/experiments/lib")
sys.path.insert(0, "/home/ender/uni/msc-thesis/playground/bargain_screen")

from corpora import DATASETS, loaded, proxy_scores, subgraph, vectors_for
from search import adjacency, knn, search

NAMES = sys.argv[1:] or ["cad", "meld", "meld_dialogue"]
SEEDS = 5
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
T, DELTA = 0.9, 0.1
RS = [22, 40, 80]
DRAWS = 300
ARMS = ["proxy", "proxy+knn", "proxy+rewired", "proxy+links"]


def rewire(edges, seed=0):
    stubs = np.random.default_rng(seed).permutation(edges.ravel()).reshape(-1, 2)
    return stubs[stubs[:, 0] != stubs[:, 1]]


def positions(job):
    """Call number at which each positive is found, in order of discovery."""
    y, p, nb, seed = job
    P = int(y.sum())
    recalls = (np.arange(1, P + 1) - 0.5) / P
    return np.array(search(y, "proxy+links", None, nb, seed, recalls, p))


def qbcb_j(r):
    """Smallest j with P(Bin(r, T) >= j) <= delta, None if even j = r fails."""
    for j in range(1, r + 1):
        if binom.sf(j - 1, r, T) <= DELTA:
            return j
    return None


def jobs_for(name):
    ds = DATASETS[name]
    nodes, edges, _, masks, pools = loaded(ds)
    mask = pools[1][1]
    n = int(mask.sum())
    links = subgraph(mask, edges)
    graphs = {
        "proxy": [np.array([], int)] * n,
        "proxy+knn": adjacency(knn(vectors_for(nodes)[mask], max(1, round(len(links) / n))), n),
        "proxy+rewired": adjacency(rewire(links), n),
        "proxy+links": adjacency(links, n),
    }
    scores = proxy_scores(ds, len(nodes))
    out = []
    for key, _, _ in ds.predicates:
        y = masks[key][mask]
        p = np.nan_to_num(scores[key][mask])
        out.append((key, y, p, graphs))
    return out


def main():
    print(f"recall target {T}, delta {DELTA}, {SEEDS} seeds, QBCB {DRAWS} sample draws")
    print("orders: proxy p(yes), with a neighbour of a found positive above every untouched row")
    print()
    for name in NAMES:
        items = jobs_for(name)
        work = [(y, p, g[a], s) for _, y, p, g in items for a in ARMS for s in range(SEEDS)]
        with ProcessPoolExecutor(8, mp_context=get_context("fork")) as pool:
            res = list(pool.map(positions, work, chunksize=1))
        it = iter(res)
        print(f"== {name}")
        print(f"{'predicate':>13} {'rate':>6} {'arm':>14} " + " ".join(f"R@{b:.0%}".rjust(7) for b in BUDGETS)
              + " " + " ".join(f"qbcb{r}".rjust(8) for r in RS) + "  met")
        for key, y, p, _ in items:
            n = len(y)
            P = int(y.sum())
            for a in ARMS:
                calls = [next(it) for _ in range(SEEDS)]
                rec = [np.mean([(c <= b * n).sum() / P for c in calls]) for b in BUDGETS]
                costs, mets = [], []
                for r in RS:
                    cs, rs = [], []
                    rng = np.random.default_rng(1)
                    for d in range(DRAWS):
                        c = calls[d % SEEDS]
                        cs_, rs_ = qbcb_calls(c, y, r, rng)
                        cs.append(cs_)
                        rs.append(rs_)
                    costs.append(np.nanmean(cs))
                    mets.append(np.nanmean(np.array(rs) >= T))
                print(f"{key:>13} {y.mean():>6.1%} {a:>14} " + " ".join(f"{x:>7.1%}" for x in rec)
                      + " " + " ".join(f"{x:>8.1%}" for x in costs) + "  " + "/".join(f"{m:.0%}" for m in mets))
        print()


def qbcb_calls(calls, y, r, rng):
    """QBCB on the discovery times. Which positive got which call does not matter for the
    sample, since sample positives are a uniform subset of positives, so draw them as a
    uniform subset of the discovery times. Negatives in the sample add cost unless screened.
    """
    j = qbcb_j(r)
    n = len(y)
    P = len(calls)
    if P < r:
        return np.nan, np.nan
    # inverse sampling: rows drawn until r positives, the negatives drawn are hypergeometric
    perm = rng.permutation(n)
    cum = np.cumsum(y[perm])
    m = int(np.searchsorted(cum, r)) + 1
    negs = m - r
    picks = rng.choice(P, r, replace=False)
    stop = np.sort(calls[picks])[j - 1]
    found = (calls <= stop).sum()
    screened_neg = stop - found
    # sampled negatives are a uniform subset of all negatives, a share of them already screened
    extra_neg = negs * (1 - screened_neg / (n - P))
    cost = (stop + extra_neg) / n
    return cost, found / P


if __name__ == "__main__":
    main()
