"""Target method stopping rule (Cormack and Grossman, Lewis et al. quantile version) on top of any oracle order.

Sample rows uniformly until K positives are found. S of them (random) are seeds the order may use, the other k are hidden targets.
Screen in the order until r of the k targets are screened. P(recall < T) <= P(Bin(k, T) >= r) <= delta.
Cost is the distinct oracle calls, sample plus screened rows, the returned set is the confirmed positives.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path("/home/ender/uni/msc-thesis")
sys.path.insert(0, str(ROOT / "experiments/lib"))
from corpora import DATASETS, loaded, proxy_scores, subgraph
from search import adjacency

T, DELTA, RUNS = 0.9, 0.1, 200


def rule(k):
    return next(r for r in range(k + 1) if binom.sf(r - 1, k, T) <= DELTA)


def order(p, y, neighbours, seeds=()):
    """Highest p first, a neighbour of a found positive above every other row, as phase1 in links.py."""
    n = len(y)
    asked = np.zeros(n, bool)
    front = np.zeros(n)
    for s in seeds:
        if neighbours is not None:
            front[neighbours[s]] = 2
    cur = p + front
    cur[list(seeds)] = -np.inf
    out = []
    for _ in range(n - len(seeds)):
        i = int(np.argmax(cur))
        out.append(i)
        cur[i] = -np.inf
        if y[i] and neighbours is not None:
            nb = neighbours[i]
            nb = nb[np.isfinite(cur[nb]) & (front[nb] == 0)]
            front[nb] = 2
            cur[nb] += 2
    return np.array(list(seeds) + out, int)


def depth90(o, y):
    c = np.cumsum(y[o])
    return (np.searchsorted(c, np.ceil(T * y.sum())) + 1) / len(y)


def run(o_fixed, p, y, nb, K, k, r, rng, seeded):
    n = len(y)
    perm = rng.permutation(n)
    c = np.cumsum(y[perm])
    m = np.searchsorted(c, K) + 1
    sample = perm[:m]
    pos = sample[y[sample]]
    rng.shuffle(pos)
    seeds, targets = pos[: K - k], pos[K - k:]
    o = order(p, y, nb, seeds) if seeded else o_fixed
    rank = np.empty(n, int)
    rank[o] = np.arange(n)
    stop = np.sort(rank[targets])[r - 1] + 1
    screened = np.zeros(n, bool)
    screened[o[:stop]] = True
    screened[sample] = True
    return screened.mean(), (y & screened).sum() / y.sum() >= T, m / n


for name in sys.argv[1:] or ["cad", "meld"]:
    ds = DATASETS[name]
    nodes, edges, _, masks, pools = loaded(ds)
    mask = pools[1][1]
    n = int(mask.sum())
    links = subgraph(mask, edges)
    nb = adjacency(links, n)
    scores = proxy_scores(ds, len(nodes))
    A = csr_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(n, n))
    print(f"\n{name}, {n} rows, {len(links)} edges, {RUNS} runs, T {T}, delta {DELTA}")
    print(f"{'predicate':>13} {'rate':>5} {'pos':>5} {'netsz>1':>7} | {'03 p':>5} {'03 p+l':>6} | "
          + " ".join(f"{f'{K}/{k}/{r} {a}':>16}" for K, k, r, a, _ in [(0, 0, 0, '', 0)][:0])
          + "k/r  arm  cost met sample")
    rng0 = np.random.default_rng(0)
    for key, _, _ in ds.predicates:
        y = masks[key][mask]
        p = np.nan_to_num(scores[key][mask]) + 1e-9 * rng0.random(n)
        Ay = A[y][:, y]
        _, lab = connected_components(Ay, directed=False)
        sizes = np.bincount(lab)
        multi = sizes[sizes > 1].sum() / y.sum()
        op = order(p, y, None)
        ol = order(p, y, nb)
        line = f"{key:>13} {y.mean():>5.1%} {y.sum():>5} {multi:>7.1%} | {depth90(op, y):>5.1%} {depth90(ol, y):>6.1%} |"
        print(line)
        for k in [22, 40, 80]:
            r = rule(k)
            for arm, o, seeded, K in [("proxy", op, False, k), ("links", ol, False, k), ("links+10seeds", ol, True, k + 10)]:
                if seeded and k != 40:
                    continue
                rng = np.random.default_rng(1)
                res = np.array([run(o, p, y, nb, K, k, r, rng, seeded) for _ in range(RUNS if not seeded else 40)])
                print(f"{'':>45} k {k:>3} r {r:>3} {arm:>14} cost {res[:, 0].mean():>6.1%} met {res[:, 1].mean():>5.1%} sample {res[:, 2].mean():>5.1%}")
