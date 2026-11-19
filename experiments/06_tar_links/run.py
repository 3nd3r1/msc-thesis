"""Verified positives at a guaranteed recall: QBCB on four oracle orders against BARGAIN plus checking."""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from BARGAIN import BARGAIN_R, Oracle, Proxy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))

from corpora import DATASETS, loaded, proxy_scores, subgraph, vectors_for
from report import report
from search import adjacency, knn

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["cad", "meld", "cora", "stack"]
TESTED = ["cad", "meld"]
T, DELTA = 0.9, 0.1
K = 40
RUNS = 2000
BUDGETS = [400, 0.01, 0.02, 0.05, 0.10]
BARGAIN_RUNS = 50
ORDERS = ["proxy", "knn", "rewired", "links"]
FLOOR = 0.88
R = next(r for r in range(1, K + 1) if binom.sf(r - 1, K, T) <= DELTA)


class Scores(Proxy):
    def __init__(self, s):
        super().__init__(verbose=False)
        self.s = s

    def get_preds_and_scores(self, indxs, data_records):
        return np.ones(len(indxs), bool), self.s[np.asarray(indxs, int)]


class Labels(Oracle):
    def __init__(self, y):
        super().__init__(verbose=False)
        self.y = y
        self.asked = set()

    def oracle_func(self, input, proxy_output):
        self.asked.add(int(input))
        return True, bool(self.y[int(input)])


def rewire(edges, seed=0):
    """Every row keeps its degree and its edges go to random rows."""
    stubs = np.random.default_rng(seed).permutation(edges.ravel()).reshape(-1, 2)
    return stubs[stubs[:, 0] != stubs[:, 1]]


def order(p, y, neighbours):
    """Highest p first, with any neighbour of a confirmed positive above every other row."""
    cur = p.copy()
    lifted = np.zeros(len(y), bool)
    out = np.empty(len(y), int)
    for step in range(len(y)):
        i = int(np.argmax(cur))
        out[step] = i
        cur[i] = -np.inf
        if y[i] and neighbours is not None:
            nb = neighbours[i][~lifted[neighbours[i]]]
            lifted[nb] = True
            cur[nb] += 2
    return out


def tar(o, y, rng):
    """Sample uniformly until K positives, hide them, screen in order o until R of them are screened."""
    n = len(y)
    perm = rng.permutation(n)
    c = np.cumsum(y[perm])
    if c[-1] < K:
        return 1.0, True
    sample = perm[: np.searchsorted(c, K) + 1]
    targets = sample[y[sample]]
    rank = np.empty(n, int)
    rank[o] = np.arange(n)
    stop = np.sort(rank[targets])[R - 1] + 1
    asked = np.zeros(n, bool)
    asked[o[:stop]] = True
    asked[sample] = True
    return asked.mean(), (y & asked).sum() >= T * y.sum()


def bargain(p, y, budget, seed):
    """BARGAIN's recall query, then the oracle checks every returned row it has not asked."""
    oracle = Labels(y)
    found = BARGAIN_R(Scores(p), oracle, DELTA, T, budget, seed=seed).process(list(range(len(y))))
    found = np.array(found, int)
    return len(oracle.asked | set(found.tolist())) / len(y), y[found].sum() >= T * y.sum()


def one(job):
    name, key, y, p, graphs = job
    n = len(y)
    row = {"dataset": name, "predicate": key, "rate": y.mean(), "positives": int(y.sum())}
    for arm, nb in graphs.items():
        o = order(p, y, nb)
        rng = np.random.default_rng(0)
        res = np.array([tar(o, y, rng) for _ in range(RUNS)])
        row[arm] = res[:, 0].mean()
        row[arm + " met"] = res[:, 1].mean()
    budgets = [b if b >= 1 else max(1, round(b * n)) for b in BUDGETS]
    for b in budgets:
        res = np.array([bargain(p, y, b, s) for s in range(BARGAIN_RUNS)])
        row[f"bargain {b}"] = res[:, 0].mean()
        row[f"bargain {b} met"] = res[:, 1].mean()
    costs = {b: row[f"bargain {b}"] for b in budgets}
    best = min(costs, key=costs.get)
    row["bargain 400"] = costs[400]
    row["bargain best"] = costs[best]
    row["best budget"] = best
    row["bargain best met"] = row[f"bargain {best} met"]
    print(name, key, flush=True)
    return row


def jobs(names):
    for name in names:
        ds = DATASETS[name]
        nodes, edges, _, masks, pools = loaded(ds)
        mask = pools[1][1]
        n = int(mask.sum())
        links = subgraph(mask, edges)
        graphs = {
            "proxy": None,
            "knn": adjacency(knn(vectors_for(nodes)[mask], max(1, round(len(links) / n))), n),
            "rewired": adjacency(rewire(links), n),
            "links": adjacency(links, n),
        }
        scores = proxy_scores(ds, len(nodes))
        tie = np.random.default_rng(0).random(n) * 1e-9
        for key, _, _ in ds.predicates:
            yield name, key, masks[key][mask], np.nan_to_num(scores[key][mask]) + tie, graphs


def main():
    names = sys.argv[1:] or NAMES
    with ProcessPoolExecutor(mp_context=get_context("fork")) as pool:
        rows = list(pool.map(one, jobs(names)))

    cols = ["bargain 400", "bargain best"] + ORDERS
    out, save = report(RESULTS, "costs" if names == NAMES else "_".join(names))
    out(f"recall target {T}, delta {DELTA}, pool with junk and repeated texts dropped")
    out(f"cost: distinct oracle calls as a share of the rows, every returned row confirmed, asking every row costs 100%")
    out(f"bargain: BARGAIN_R on the 8B p(yes), then the oracle checks the returned rows, mean of {BARGAIN_RUNS} runs")
    out(f"bargain 400: the paper's budget, bargain best: lowest cost over the budgets {BUDGETS}, picked in hindsight")
    out(f"proxy, knn, rewired, links: QBCB with k {K} and r {R}, mean of {RUNS} runs, the order is the 8B p(yes)")
    out("with any neighbour of a confirmed positive above every other row, over no graph, a cosine kNN with")
    out("k matched to the mean link degree, the links rewired with every degree kept, or the links")
    out("met: lowest share of runs reaching the recall target over the arms of the row")
    out()
    out(f"{'dataset':>8} {'predicate':>13} {'rate':>6} {'pos':>5} " + " ".join(f"{c:>12}" for c in cols) + f" {'met':>5}")
    for r in rows:
        met = min(r[c + " met"] for c in cols)
        out(f"{r['dataset']:>8} {r['predicate']:>13} {r['rate']:>6.1%} {r['positives']:>5} "
            + " ".join(f"{r[c]:>12.1%}" for c in cols) + f" {met:>5.0%}")
    out()

    out("links minus each other arm, mean over predicates, in points of the rows")
    out()
    others = ["bargain best", "proxy", "knn", "rewired"]
    out(f"{'dataset':>8} " + " ".join(f"{c:>12}" for c in others))
    diffs = {}
    for name in names:
        sub = [r for r in rows if r["dataset"] == name]
        diffs[name] = {c: 100 * np.mean([r["links"] - r[c] for r in sub]) for c in others}
        out(f"{name:>8} " + " ".join(f"{diffs[name][c]:>+12.2f}" for c in others))
    out()

    if names == NAMES:
        valid = all(r[a + " met"] >= FLOOR for r in rows for a in ORDERS)
        passed = all(diffs[name][c] < 0 for name in TESTED for c in others)
        out(f"every QBCB cell meets the target in at least {FLOOR:.0%} of runs: {valid}")
        out(f"links below every other arm on {' and '.join(TESTED)}: {passed}")
        out(f"verdict: {'invalid' if not valid else 'pass' if passed else 'fail'}")
    save()


if __name__ == "__main__":
    main()
