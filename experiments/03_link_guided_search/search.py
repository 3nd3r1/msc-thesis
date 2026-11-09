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
    degree = np.array([len(x) for x in neighbours], float)
    found = 0

    # Scores of the rows in `ix`, so a step can rescore only the rows it changed.
    score = {
        "random": lambda ix: tie[ix],
        "embeddings": lambda ix: mean[ix],
        "links": lambda ix: pos[ix],
        "links+embeddings": lambda ix: pos[ix] + mean[ix],
        "links+-": lambda ix: pos[ix] - neg[ix],
        "degree": lambda ix: degree[ix],
        "blend/deg": lambda ix: (pos[ix] - neg[ix]) / np.maximum(degree[ix], 1) + mean[ix],
        "proxy": lambda ix: prior[ix],
        # A neighbour of a confirmed positive outranks every row that is not
        # one, and the proxy orders within each of the two sets.
        "proxy+links": lambda ix: prior[ix] + 2.0 * (pos[ix] > 0),
    }.get(order, lambda ix: float(order.split()[1]) * (pos[ix] - neg[ix]) + mean[ix])
    # The matvec behind `mean` is almost the whole cost, so skip it where unread.
    similar = order.split()[0] in ("embeddings", "links+embeddings", "blend", "blend/deg")

    def rescore(ix):
        current[ix] = np.where(asked[ix], -np.inf, score(ix) + 1e-9 * tie[ix])

    current = np.zeros(n)
    rescore(slice(None))
    for calls in range(1, n + 1):
        i = int(np.argmax(current))
        asked[i] = True
        current[i] = -np.inf
        if labels[i]:
            found += 1
            pos[neighbours[i]] += 1
            if similar:
                similarity += vectors @ vectors[i]
                mean = similarity / found
                rescore(slice(None))
            else:
                rescore(neighbours[i])
            while len(hit) < len(targets) and found >= targets[len(hit)]:
                hit.append(calls)
            if len(hit) == len(targets):
                return hit
        else:
            neg[neighbours[i]] += 1
            rescore(neighbours[i])
    return hit + [n] * (len(targets) - len(hit))
