import numpy as np

DECAY = 0.99


def adjacency(edges, n):
    out = [[] for _ in range(n)]
    for a, b in edges:
        out[a].append(b)
        out[b].append(a)
    return [np.array(x, dtype=int) for x in out]


def unit(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-9)


def search(labels, order, vectors, neighbours, seed=0, recalls=(0.90,), proxy=None,
           decay=DECAY, head=0.0):
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
        "proxy+switch": lambda ix: prior[ix],
    }.get(order, lambda ix: float(order.split()[1]) * (pos[ix] - neg[ix]) + mean[ix])
    # The matvec behind `mean` is almost the whole cost, so skip it where unread.
    similar = order.split()[0] in ("embeddings", "links+embeddings", "blend", "blend/deg")

    def rescore(ix):
        current[ix] = np.where(asked[ix], -np.inf, score(ix) + 1e-9 * tie[ix])

    # proxy+switch picks between the proxy's best row and the proxy's best
    # neighbour of a confirmed positive by Thompson sampling on how often each
    # source has hit. Counts decay since the proxy's hit rate falls as it goes.
    # `head` is pseudo hits for the proxy, so links have to earn their turn.
    hits = np.array([head, 0.0])
    misses = np.zeros(2)

    current = np.zeros(n)
    rescore(slice(None))
    for calls in range(1, n + 1):
        i = int(np.argmax(current))
        if order == "proxy+switch":
            near = np.where(pos > 0, current, -np.inf)
            j = int(np.argmax(near))
            if near[j] > -np.inf and j != i:
                source = int(np.argmax(rng.beta(hits + 1, misses + 1)))
                i = (i, j)[source]
                hits *= decay
                misses *= decay
                hits[source] += labels[i]
                misses[source] += 1 - labels[i]
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
