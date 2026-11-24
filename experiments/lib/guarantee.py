"""Verified positives at a recall guarantee: oracle orders, QBCB, BARGAIN plus checking and a cluster bootstrap."""

import numpy as np
from BARGAIN import BARGAIN_R, Oracle, Proxy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression


def stop_count(k, target, delta):
    """Smallest r with P(Bin(k, target) >= r) <= delta."""
    return next(r for r in range(1, k + 1) if binom.sf(r - 1, k, target) <= delta)


def rewire(edges, seed=0):
    """Every row keeps its degree and its edges go to random rows."""
    stubs = np.random.default_rng(seed).permutation(edges.ravel()).reshape(-1, 2)
    return stubs[stubs[:, 0] != stubs[:, 1]]


def knn_draw(vectors, rows, k, block=2048):
    """Top k by cosine among the given rows, never to another copy of the same row."""
    v = vectors[rows]
    pairs = set()
    for s in range(0, len(rows), block):
        sim = v[s:s + block] @ v.T
        sim[rows[s:s + block, None] == rows[None, :]] = -2
        for r, top in enumerate(np.argpartition(-sim, k, axis=1)[:, :k]):
            i = s + r
            pairs.update((min(i, int(j)), max(i, int(j))) for j in top)
    return np.array(sorted(pairs))


def order(p, y, neighbours=None):
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


def cal(p, y, X, neighbours=None, batch=0.01):
    """Continuous active learning, the proxy order until both classes are seen, then a logistic
    regression refit on the verdicts after every batch, with counts of confirmed positive and
    negative neighbours as two more features when there are neighbours."""
    n = len(y)
    b = max(1, round(batch * n))
    logit = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    asked = np.zeros(n, bool)
    pos = np.zeros(n)
    neg = np.zeros(n)
    score = p.copy()
    out = []
    while len(out) < n:
        free = np.flatnonzero(~asked)
        pick = free[np.argsort(-score[free], kind="stable")[:b]]
        out.extend(pick.tolist())
        asked[pick] = True
        if neighbours is not None:
            for i in pick:
                (pos if y[i] else neg)[neighbours[i]] += 1
        if y[asked].all() or not y[asked].any() or asked.all():
            continue
        cols = [logit, X] + ([np.log1p(pos), np.log1p(neg)] if neighbours is not None else [])
        F = np.column_stack(cols)
        score = LogisticRegression(max_iter=1000).fit(F[asked], y[asked]).decision_function(F)
    return np.array(out)


def qbcb(o, y, k, r, target, rng):
    """Sample uniformly until k positives, hide them, screen in order o until r of them are screened.
    Returns the share of rows asked and whether the confirmed positives reach the target."""
    n = len(y)
    perm = rng.permutation(n)
    c = np.cumsum(y[perm])
    if c[-1] < k:
        return 1.0, True
    sample = perm[: np.searchsorted(c, k) + 1]
    targets = sample[y[sample]]
    rank = np.empty(n, int)
    rank[o] = np.arange(n)
    stop = np.sort(rank[targets])[r - 1] + 1
    asked = np.zeros(n, bool)
    asked[o[:stop]] = True
    asked[sample] = True
    return asked.mean(), (y & asked).sum() >= target * y.sum()


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


def bargain(p, y, budget, target, delta, seed):
    """BARGAIN's recall query, then the oracle checks every returned row it has not asked."""
    oracle = Labels(y)
    found = BARGAIN_R(Scores(p), oracle, delta, target, budget, seed=seed).process(list(range(len(y))))
    found = np.array(found, int)
    return len(oracle.asked | set(found.tolist())) / len(y), y[found].sum() >= target * y.sum()


def clusters(edges, n):
    """Connected components of the links, the threads on CAD and the dialogues on MELD."""
    A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(A, directed=False)[1]


def resample(group, rng):
    """A cluster bootstrap, whole clusters drawn with replacement.
    Returns the original row of every new row and which draw it came from."""
    by = np.argsort(group, kind="stable")
    members = np.split(by, np.cumsum(np.bincount(group))[:-1])
    draw = rng.integers(len(members), size=len(members))
    rows = np.concatenate([members[g] for g in draw])
    copy = np.repeat(np.arange(len(draw)), [len(members[g]) for g in draw])
    return rows, copy


def remap(edges, rows, copy, within=True):
    """Edges among the resampled rows. Within a cluster each draw gets its own copy of the edges,
    otherwise every copy of a row links to the first copy of its neighbour."""
    key = {(int(c), int(r)): i for i, (c, r) in enumerate(zip(copy, rows))}
    copies = {}
    for i, (c, r) in enumerate(zip(copy, rows)):
        copies.setdefault(int(r), []).append((int(c), i))
    out = []
    for a, b in edges:
        for c, i in copies.get(int(a), []):
            if within and (c, int(b)) in key:
                out.append((i, key[c, int(b)]))
            elif not within and int(b) in copies:
                out.append((i, copies[int(b)][0][1]))
    return np.array(out, int).reshape(-1, 2)
