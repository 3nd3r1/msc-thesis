import numpy as np


def adjacency(edges, n):
    out = [[] for _ in range(n)]
    for a, b in edges:
        out[a].append(b)
        out[b].append(a)
    return [np.array(x, dtype=int) for x in out]


def unit(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-9)


def search(labels, order, vectors, neighbours, seed=0, recalls=(0.90,), proxy=None):
    """Oracle calls until each of `recalls` of the positives are found, lowest first."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    targets = [int(np.ceil(r * labels.sum())) for r in recalls]
    hit = []
    tie = rng.random(n)
    pos = np.zeros(n)
    neg = np.zeros(n)
    similarity = np.zeros(n)
    mean = np.zeros(n)
    prior = np.zeros(n) if proxy is None else proxy
    asked = np.zeros(n, bool)
    found = 0

    # Lazy, so one search evaluates one order rather than all of them.
    score = {
        "random": lambda: tie,
        "embeddings": lambda: mean,
        "links": lambda: pos,
        "links+embeddings": lambda: pos + mean,
        "links+-": lambda: pos - neg,
        "proxy": lambda: prior,
        # A neighbour of a confirmed positive outranks every row that is not
        # one, and the proxy orders within each of the two sets.
        "proxy+links": lambda: prior + 2.0 * (pos > 0),
    }.get(order, lambda: float(order.split()[1]) * (pos - neg) + mean)
    # The matvec behind `mean` is almost the whole cost, so skip it where unread.
    similar = order.split()[0] in ("embeddings", "links+embeddings", "blend")

    for calls in range(1, n + 1):
        current = score() + 1e-9 * tie
        current[asked] = -np.inf

        i = int(np.argmax(current))
        asked[i] = True
        if labels[i]:
            found += 1
            if similar:
                similarity += vectors @ vectors[i]
                mean = similarity / found
            pos[neighbours[i]] += 1
            while len(hit) < len(targets) and found >= targets[len(hit)]:
                hit.append(calls)
            if len(hit) == len(targets):
                return hit
        else:
            neg[neighbours[i]] += 1
    return hit + [n] * (len(targets) - len(hit))
