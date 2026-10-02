import numpy as np
import pandas as pd

from metrics import score


def indices(keys):
    codes = pd.factorize(keys)[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes))[:-1]
    return np.split(order, bounds)


def majority(truth):
    return np.full(len(truth), truth.mean() > 0.5)


def replay(groups, truth, k, threshold, rng):
    if k == 0:
        return 0, majority(truth)

    guess = np.empty(len(truth), dtype=bool)
    calls = 0
    for idx in groups:
        if len(idx) <= k:
            guess[idx] = truth[idx]
            calls += len(idx)
            continue

        seen = rng.choice(idx, k, replace=False)
        rest = np.setdiff1d(idx, seen, assume_unique=True)
        guess[seen] = truth[seen]
        calls += k

        yes = truth[seen].mean()
        if max(yes, 1 - yes) >= threshold:
            guess[rest] = yes > 0.5
        else:
            guess[rest] = truth[rest]
            calls += len(rest)
    return calls, guess


def sweep(groups, truth, configs, repeats, rng):
    results = [{"k": None, "threshold": 1.0, "calls": 1.0, "accuracy": 1.0, "f1": 1.0}]
    for k, threshold in configs:
        runs = [replay(groups, truth, k, threshold, rng) for _ in range(repeats)]
        accuracy, f1 = np.mean([score(truth, g) for _, g in runs], axis=0)
        results.append(
            {
                "k": k,
                "threshold": threshold,
                "calls": np.mean([c for c, _ in runs]) / len(truth),
                "accuracy": accuracy,
                "f1": f1,
            }
        )
    return results


def cheapest(results, target):
    return min(
        (r for r in results if r["accuracy"] >= target), key=lambda r: r["calls"]
    )
