import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import report
from search import adjacency, search, unit

from data import load_cora


RESULTS = Path(__file__).resolve().parent / "results"

SEEDS = 5
RECALL = 0.90
ORDERS = ["random", "embeddings", "links", "links+embeddings"]


def main():
    nodes, features, edges = load_cora()
    vectors = unit(features)
    neighbours = adjacency(edges, len(nodes))
    out, save = report(RESULTS, "cora")

    out(f"cora, {len(nodes):,} papers, {len(edges):,} citations")
    out(f"calls to {RECALL:.0%} recall, mean of {SEEDS} seeds")
    out()
    out("  calls% +- sd" + "".join(f"{o:>22}" for o in ORDERS))

    for label in nodes["label"].value_counts().index:
        labels = (nodes["label"] == label).to_numpy()
        line = f"{label:>24} {labels.mean():>5.1%}"
        for order in ORDERS:
            calls = [
                search(labels, order, vectors, neighbours, s, RECALL)
                for s in range(SEEDS)
            ]
            share = np.array(calls) / len(nodes)
            line += f"   {share.mean():>6.1%}+-{share.std():.1%}"
        out(line)

    save()


if __name__ == "__main__":
    main()
