from collections import deque

import numpy as np
import pandas as pd

from embed import kmeans


def indices(keys):
    codes = pd.factorize(keys)[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes))[:-1]
    return np.split(order, bounds)


def sample_size(n, fraction, floor):
    return max(int(np.ceil(fraction * n)), floor)


def split(idx, vectors, seed):
    if len(idx) < 2:
        return [idx]
    labels = kmeans(vectors[idx], 2, seed)
    parts = [idx[labels == c] for c in (0, 1)]
    return [p for p in parts if len(p)]


def run(groups, truth, vectors, target, rng, seed, fraction, floor):
    guess = np.empty(len(truth), dtype=bool)
    calls = 0
    found = 0
    trace = []
    queue = deque(groups)

    while queue:
        idx = queue.popleft()
        size = sample_size(len(idx), fraction, floor)

        if len(idx) <= size:
            guess[idx] = truth[idx]
            calls += len(idx)
            found += truth[idx].sum()
            trace.append((calls, found))
            continue

        seen = rng.choice(idx, size, replace=False)
        rest = np.setdiff1d(idx, seen, assume_unique=True)
        guess[seen] = truth[seen]
        calls += size
        found += truth[seen].sum()

        yes = truth[seen].mean()
        if max(yes, 1 - yes) >= target:
            guess[rest] = yes > 0.5
            if yes > 0.5:
                found += truth[rest].sum()
        else:
            parts = split(rest, vectors, seed)
            if len(parts) == 2:
                queue.extend(parts)
                trace.append((calls, found))
                continue
            guess[rest] = truth[rest]
            calls += len(rest)
            found += truth[rest].sum()

        trace.append((calls, found))

    return calls, guess, trace


def calls_at_recall(trace, positives, target):
    need = target * positives
    for calls, found in trace:
        if found >= need:
            return calls
    return None
