import os
import re
import sys
from pathlib import Path

# One BLAS thread per process, set before numpy loads, see 03.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from corpora import DATASETS, loaded, proxy_scores, subgraph, vectors_for
from report import report
from search import SEEDS, adjacency, knn, table

RESULTS = Path(__file__).resolve().parent / "results"
HELP = ["cad", "meld", "meld_dialogue"]
CONTROL = ["cora", "stack", "stack_author"]
GRAPHS = ["links", "knn", "both"]
ORDERS = ["proxy", "proxy+diffuse"]


def run(ds):
    nodes, edges, _, masks, pools = loaded(ds)
    scores = proxy_scores(ds, len(nodes))
    title, mask = pools[1]
    n = int(mask.sum())
    vectors = vectors_for(nodes)[mask]
    links = subgraph(mask, edges)
    cos = knn(vectors, max(1, round(len(links) / n)))
    graphs = {"links": links, "knn": cos, "both": np.unique(np.r_[links, cos], axis=0)}
    names = [
        (f"{k} {kind}", masks[k][mask], rankdata(scores[k][mask]) / n)
        for k, kind, _ in ds.predicates
    ]

    out, save = report(RESULTS, ds.name)
    out(f"{ds.name}, {title}")
    out(f"calls% is calls over rows, mean and sd of {SEEDS} seeds")
    out("the proxy score is the rank of the 8B's p over the rows, ties share their mean rank")
    out("proxy+diffuse adds the mean residual, verdict minus rank, of the asked neighbours")
    out("knn is the top k cosine neighbours, k picked so it has about as many edges as the links")
    for graph, pairs in graphs.items():
        out(f"\n{graph}, {len(pairs):,} edges")
        cache = RESULTS / f"{ds.name}.{graph}.searches.jsonl"
        table(out, names, vectors, adjacency(pairs, n), cache, ORDERS)
        cache.unlink()
    save()


def gaps(name):
    """proxy+diffuse minus proxy at 90% recall per graph, in points of the rows."""
    text = (RESULTS / f"{name}.txt").read_text()
    out = {}
    for graph in GRAPHS:
        block = text.split(f"\n{graph}, ")[1].split("90% recall")[1].split("recall")[0]
        rows = re.findall(r"([\d.]+)%\+-[\d.]+% +([\d.]+)%\+-[\d.]+%", block)
        out[graph] = [float(b) - float(a) for a, b in rows]
    return out


def verdict():
    out, save = report(RESULTS, "verdict")
    out("proxy+diffuse minus proxy at 90% recall, mean over predicates, in points of the rows")
    out(f"\n{'dataset':>14}" + "".join(f"{g:>8}" for g in GRAPHS))
    pooled = {g: [] for g in GRAPHS}
    for name in HELP + CONTROL:
        g = gaps(name)
        out(f"{name:>14}" + "".join(f"{np.mean(g[x]):>+8.1f}" for x in GRAPHS))
        if name in HELP:
            for x in GRAPHS:
                pooled[x] += g[x]
    links, cos = np.mean(pooled["links"]), np.mean(pooled["knn"])
    out(f"\nover {', '.join(HELP)}: links {links:+.2f}, knn {cos:+.2f}")
    out("pass rule: links below 0 and below knn")
    out("pass" if links < 0 and links < cos else "fail")
    save()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["verdict"]:
        verdict()
    elif len(args) == 1 and args[0] in HELP + CONTROL:
        run(DATASETS[args[0]])
    else:
        sys.exit(f"usage: run.py [{'|'.join(HELP + CONTROL)}|verdict]")
