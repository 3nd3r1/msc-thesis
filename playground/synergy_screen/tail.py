"""Of the positives the embedding order finds last, how many cite or are cited by one it found earlier."""

from pathlib import Path

import numpy as np

from pick import picked
from report import report
from screen import REPO, citations, embedded, pool
from search import adjacency

SEEDS = 10
HEAD, TAIL = 0.80, 0.95


def found(labels, vectors, seed):
    """Positives and the call each was found on, in the embeddings order of 03, up to TAIL."""
    tie = 1e-9 * np.random.default_rng(seed).random(len(labels))
    stop = int(np.ceil(TAIL * labels.sum()))
    similarity = np.zeros(len(labels))
    asked = np.zeros(len(labels), bool)
    rows, calls = [], []
    for call in range(1, len(labels) + 1):
        score = (similarity / len(rows) if rows else 0) + tie
        score[asked] = -np.inf
        i = int(np.argmax(score))
        asked[i] = True
        if labels[i]:
            rows.append(i)
            calls.append(call)
            similarity += vectors @ vectors[i]
            if len(rows) == stop:
                break
    return rows, calls


if __name__ == "__main__":
    out, save = report(Path(__file__).resolve().parent / "results", "tail")
    out(f"embeddings order, mean of {SEEDS} seeds")
    out(f"tail is the positives found between {HEAD:.0%} and {TAIL:.0%} recall")
    out("linked: tail positives with a citation edge to a positive found before them")
    out("cited: tail positives with a citation edge to any positive")
    out("calls: share of rows asked to reach each target, and the share spent on the tail")
    out()
    out(f"{'review':>20} {'pos':>5} {'tail':>5} {'linked':>7} {'cited':>6} "
        f"{HEAD:>6.0%} {TAIL:>6.0%} {'spent':>6}")

    for name, n, pos, _ in picked():
        data = pool(REPO / name / f"{name}_ids.csv")
        labels = np.array([bool(r["included"]) for r in data])
        neighbours = adjacency(citations(data), n)
        vectors = embedded(data)
        head = int(np.ceil(HEAD * pos))

        linked, cited, at_head, at_tail = [], [], [], []
        for seed in range(SEEDS):
            rows, calls = found(labels, vectors, seed)
            tail = range(head, len(rows))
            linked.append(np.mean([np.isin(neighbours[rows[k]], rows[:k]).any() for k in tail]))
            cited.append(np.mean([labels[neighbours[rows[k]]].any() for k in tail]))
            at_head.append(calls[head - 1] / n)
            at_tail.append(calls[-1] / n)

        out(f"{name:>20} {pos:>5} {len(tail):>5} {np.mean(linked):>7.1%} {np.mean(cited):>6.1%} "
            f"{np.mean(at_head):>6.1%} {np.mean(at_tail):>6.1%} "
            f"{np.mean(at_tail) - np.mean(at_head):>6.1%}")

    save()
