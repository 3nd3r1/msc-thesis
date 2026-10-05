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


def search(labels, order, vectors, neighbours, seed=0, recall=0.90):
    """Oracle calls until `recall` of the positives are found."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    target = int(np.ceil(recall * labels.sum()))
    tie = rng.random(n)
    link = np.zeros(n)
    similarity = np.zeros(n)
    asked = np.zeros(n, bool)
    found = 0

    for calls in range(1, n + 1):
        mean = similarity / max(found, 1)
        score = {
            "random": tie,
            "embeddings": mean,
            "links": link,
            "links+embeddings": link + mean,
        }[order] + 1e-9 * tie
        score[asked] = -np.inf

        i = int(np.argmax(score))
        asked[i] = True
        if labels[i]:
            found += 1
            similarity += vectors @ vectors[i]
            link[neighbours[i]] += 1
            if found >= target:
                return calls
    return n
