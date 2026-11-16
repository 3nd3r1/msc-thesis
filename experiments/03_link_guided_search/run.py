import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# One BLAS thread per process, set before numpy loads. Without it the workers
# below each start a thread team and fight over the cores, 20x slower.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import ORACLE, PROXY, SERVED, SYSTEM, client, judge
from report import report
from search import SEEDS, RECALLS, adjacency, knn, table, unit
from corpora import (
    CAD_SOURCE,
    DATASETS,
    HUMAN_SOURCES,
    LABELS,
    LINE_ALONE,
    MELD_SOURCE,
    MODERATION,
    STRICT,
    correlation,
    loaded,
    proxy_scores,
    subgraph,
    vectors_for,
)
from data import cora_features, load_cora


RESULTS = Path(__file__).resolve().parent / "results"

BLENDS = (1.0, 0.3, 0.1)
ORDERS = ["random", "embeddings", "links", "links+embeddings", "links+-"] + [
    f"blend {a}" for a in BLENDS
] + ["blend/deg", "degree"]
CASCADE_ORDERS = ["proxy", "proxy+links", "proxy+switch"]

WORKERS = 24
RETRIES = 8
FOLDS = 5
SPLITS = 10


def results(ds):
    return RESULTS / ds.name


def searches(ds, step):
    """Searches of a step that has not finished yet, deleted once its table is written."""
    return results(ds) / f"{step}.searches.jsonl"


def classes(ds):
    """Cora's own class labels as the oracle. The positive control."""
    if ds.name != "cora":
        sys.exit("classes is the cora control, the other datasets have no classes")

    nodes, edges = load_cora()
    out, save = report(results(ds), "classes")

    out(f"cora, {len(nodes):,} papers, {len(edges):,} citations")
    recalls = [0.90]
    out(f"calls to {recalls[0]:.0%} recall, mean of {SEEDS} seeds")
    out()

    names = [
        (label, (nodes["label"] == label).to_numpy(), None)
        for label in nodes["label"].value_counts().index
    ]
    table(
        out,
        names,
        unit(cora_features()),
        adjacency(edges, len(nodes)),
        searches(ds, "classes"),
        ORDERS,
        recalls=recalls,
    )
    save()
    searches(ds, "classes").unlink()


def collect(ds, model, name, system):
    """Ask `model` every predicate on every row, appending to labels/<ds>/<name>.jsonl."""
    nodes, _ = ds.load()
    path = LABELS / ds.labels / f"{name}.jsonl"

    done = set()
    if path.exists():
        for line in path.open():
            d = json.loads(line)
            done.add((d["row"], d["predicate"]))

    jobs = [
        (i, key, claim)
        for i in range(len(nodes))
        for key, _, claim in ds.predicates
        if (i, key) not in done
    ]
    print(
        f"{len(nodes):,} rows, {len(ds.predicates)} predicates, "
        f"{len(jobs):,} calls to make"
    )
    if done:
        print(f"resuming, {len(done):,} already in {path.name}")

    api = client()

    def work(job):
        i, key, claim = job
        row = nodes.loc[i]
        verdict = p = None
        # 429 model-busy is routine under load, so retry rather than kill the run.
        for attempt in range(RETRIES):
            try:
                verdict, p = judge(
                    api,
                    model,
                    claim,
                    {"title": row.title, "text": row.text},
                    system=system,
                )
                break
            except Exception:
                time.sleep(2**attempt)
        return {"row": i, "predicate": key, "verdict": verdict, "p": p}

    failed = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(WORKERS) as pool, path.open("a", buffering=1) as f:
        for n, rec in enumerate(pool.map(work, jobs), 1):
            if rec["verdict"] is None:
                failed += 1
            else:
                f.write(json.dumps(rec) + "\n")
            if n % 200 == 0:
                print(f"  {n:,}/{len(jobs):,}")

    if failed:
        print(f"{failed:,} calls returned no verdict")

    out, save = report(results(ds), name)
    labels = pd.read_json(path, lines=True)
    out(f"{model} on {labels.row.nunique():,} {ds.noun}, {len(labels):,} labels")
    out("served by " + (", ".join(sorted(SERVED)) or "unknown, no calls made this run"))
    out(
        {
            STRICT: "strict prompt, the claim has to be stated explicitly",
            MODERATION: "plain LOTUS prompt framed as content moderation",
            LINE_ALONE: "plain LOTUS prompt, then the calls with no verdict told to judge the line alone",
        }.get(system, "plain LOTUS prompt")
    )

    out(f"\n  {'yes':>6} {'mean p':>7} {'p<.5':>6}  predicate")
    for key, _, _ in ds.predicates:
        sub = labels[labels.predicate == key]
        if len(sub):
            yes = (sub.verdict == "yes").mean()
            out(
                f"  {yes:>6.1%} {sub.p.mean():>7.3f} {(sub.p < 0.5).mean():>6.1%}  {key}"
            )

    out(f"\n{labels.p.isna().mean():.1%} of labels have no p, logprobs were dropped")
    save()


def label(ds):
    if ds.source in HUMAN_SOURCES:
        sys.exit(f"{ds.name} has human labels, run: run.py {ds.name} human")
    collect(ds, ORACLE, "labels", STRICT)


def human(ds):
    """The human labels in the oracle's format, so every replay step reads them the same."""
    if ds.source not in HUMAN_SOURCES:
        sys.exit(f"{ds.name} has no human labels, run: run.py {ds.name} label")
    nodes, _ = ds.load()
    path = LABELS / ds.labels / "labels.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for key, _, _ in ds.predicates:
            for i, yes in enumerate(nodes[key]):
                verdict = "yes" if yes else "no"
                f.write(json.dumps({"row": i, "predicate": key, "verdict": verdict}) + "\n")
            print(f"{key}: {int(nodes[key].sum()):,} of {len(nodes):,} yes")
    print(f"wrote {path}")


def proxy(ds):
    # Human labels have no oracle prompt to match, so the proxy gets plain LOTUS.
    if ds.source == MELD_SOURCE:
        collect(ds, PROXY, "proxy", SYSTEM)
        system = LINE_ALONE
    elif ds.source == CAD_SOURCE:
        system = MODERATION
    elif ds.source in HUMAN_SOURCES:
        system = SYSTEM
    else:
        system = STRICT
    collect(ds, PROXY, "proxy", system)


def predicates(ds):
    """Replay every order on the oracle labels. No oracle calls."""
    nodes, edges, labels, masks, pools = loaded(ds)
    vectors = vectors_for(nodes)
    out, save = report(results(ds), "predicates")

    out(f"{ds.name}, {len(nodes):,} {ds.noun}, {len(edges):,} edges")
    out(f"{len(ds.predicates)} predicates labelled by {ds.source}")
    out(f"calls% is calls over rows, mean and sd of {SEEDS} seeds")
    out("embeddings are all-MiniLM-L6-v2 sentence vectors, not step 1's word features")

    counts = labels.groupby("predicate").size()
    gaps = len(nodes) - counts.reindex([k for k, _, _ in ds.predicates], fill_value=0)
    if gaps.any():
        short = ", ".join(f"{k} {v:,}" for k, v in gaps[gaps > 0].items())
        out(f"\nunlabelled rows, counted as no: {short}")

    for title, mask in pools:
        out(f"\n{title}")
        links = subgraph(mask, edges)
        names = [(f"{k} {kind}", masks[k][mask], None) for k, kind, _ in ds.predicates]
        table(
            out,
            names,
            vectors[mask],
            adjacency(links, int(mask.sum())),
            searches(ds, "predicates"),
            ORDERS,
        )

    save()
    searches(ds, "predicates").unlink()


def ranked(scores, labels, recalls):
    """Calls to each recall target when the rows go to the oracle in score order."""
    found = np.cumsum(labels[np.argsort(-scores)])
    targets = [int(np.ceil(r * labels.sum())) for r in recalls]
    return [int(np.searchsorted(found, t)) + 1 for t in targets]


def ceiling(ds):
    """What a learned order converges to once it has every label. No oracle calls."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    nodes, _, _, masks, pools = loaded(ds)
    vectors = vectors_for(nodes)
    out, save = report(results(ds), "ceiling")

    out(f"{ds.name}, {len(nodes):,} {ds.noun}")
    out(f"{len(ds.predicates)} predicates labelled by {ds.source}")
    out(f"calls% is calls over rows, mean and sd of {SPLITS} fold splits")
    out(f"logistic regression on the embeddings, {FOLDS} folds, scored out of fold")
    out("it trains on every label, so no order that learns from verdicts can do better")

    for title, mask in pools:
        out(f"\n{title}")
        v = vectors[mask]
        out(
            f"  {'rate':>6}"
            + "".join(f"{r:>15.0%}" for r in RECALLS)
            + "  predicate"
        )
        for key, kind, _ in ds.predicates:
            labels = masks[key][mask]
            runs = []
            for seed in range(SPLITS):
                score = np.zeros(len(labels))
                folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
                for train, test in folds.split(v, labels):
                    fit = LogisticRegression(max_iter=1000)
                    fit.fit(v[train], labels[train])
                    score[test] = fit.decision_function(v[test])
                runs.append(ranked(score, labels, RECALLS))
            runs = np.array(runs) / len(labels)
            line = f"  {labels.mean():>6.1%}"
            for j in range(len(RECALLS)):
                line += f"   {runs[:, j].mean():>6.1%}+-{runs[:, j].std():.1%}"
            out(line + f"  {key} {kind}")

    save()


def lift(ds):
    """How much the edges correlate the positives. No oracle calls."""
    nodes, edges, _, masks, pools = loaded(ds)
    out, save = report(results(ds), "lift")

    out(f"{ds.name}, {len(nodes):,} {ds.noun}, {len(edges):,} edges")
    out(f"{len(ds.predicates)} predicates labelled by {ds.source}")
    out()
    out("edge%: of the edges leaving a positive, the share landing on a positive")
    out("lift: edge% over the positive rate, so 1.0x is no correlation")
    out("nbr+: positives with at least one positive neighbour, which grows with degree")
    out("alone: positives with no neighbour at all, which links can never reach")

    for title, mask in pools:
        out(f"\n{title}")
        neighbours = adjacency(subgraph(mask, edges), int(mask.sum()))
        out(
            f"  {'rate':>6} {'edge%':>6} {'lift':>6} {'nbr+':>6} {'alone':>6}"
            "  predicate"
        )
        for key, kind, _ in ds.predicates:
            m = masks[key][mask]
            c = correlation(m, neighbours)
            out(
                f"  {m.mean():>6.1%} {c['edge']:>6.1%} {c['lift']:>5.1f}x "
                f"{c['nbr+']:>6.1%} {c['alone']:>6.1%}"
                f"  {key} {kind}"
            )

    save()


def cascade(ds):
    """Oracle calls once the proxy has scored every row, which both orders pay."""
    nodes, edges, _, masks, pools = loaded(ds)
    scores = proxy_scores(ds, len(nodes))
    vectors = vectors_for(nodes)
    out, save = report(results(ds), "cascade")

    out(f"{ds.name}, {len(nodes):,} {ds.noun}, {len(edges):,} edges")
    out(f"calls% is calls over rows, mean and sd of {SEEDS} seeds")
    out(f"proxy is p(yes) from {PROXY}, one call per row and predicate")
    out("both orders pay that, so only the oracle calls below differ")
    out("proxy+links sends the oracle to the best proxy score among the unasked")
    out(
        "neighbours of a confirmed positive, and to the global best when there are none"
    )
    out("proxy+switch picks between the two by Thompson sampling on which has hit more")

    for title, mask in pools:
        out(f"\n{title}")
        links = subgraph(mask, edges)
        names = [
            (f"{k} {kind}", masks[k][mask], scores[k][mask])
            for k, kind, _ in ds.predicates
        ]
        table(
            out,
            names,
            vectors[mask],
            adjacency(links, int(mask.sum())),
            searches(ds, "cascade"),
            CASCADE_ORDERS,
        )

    save()
    searches(ds, "cascade").unlink()


def diffuse(ds):
    """proxy+diffuse over links, over a cosine kNN with as many edges, and over both."""
    nodes, edges, _, masks, pools = loaded(ds)
    scores = proxy_scores(ds, len(nodes))
    vectors = vectors_for(nodes)
    out, save = report(results(ds), "diffuse")

    out(f"{ds.name}, {len(nodes):,} {ds.noun}, {len(edges):,} edges")
    out(f"calls% is calls over rows, mean and sd of {SEEDS} seeds")
    out(f"proxy is p(yes) from {PROXY}, one call per row and predicate")
    out("proxy+diffuse adds to the proxy the mean residual, verdict minus proxy,")
    out("of the asked neighbours, one hop of Correct and Smooth bias diffusion")
    out("knn is the top k cosine neighbours, k picked so it has about as many edges as the links")

    for title, mask in pools:
        n = int(mask.sum())
        links = subgraph(mask, edges)
        cos = knn(vectors[mask], max(1, round(len(links) / n)))
        both = np.unique(np.r_[links, cos], axis=0)
        names = [
            (f"{k} {kind}", masks[k][mask], scores[k][mask])
            for k, kind, _ in ds.predicates
        ]
        for graph, pairs in [("links", links), ("knn", cos), ("both", both)]:
            out(f"\n{title}, {graph}, {len(pairs):,} edges")
            table(
                out,
                names,
                vectors[mask],
                adjacency(pairs, n),
                searches(ds, f"diffuse_{graph}"),
                ["proxy", "proxy+diffuse"],
            )

    save()
    for graph in ("links", "knn", "both"):
        searches(ds, f"diffuse_{graph}").unlink()


STEPS = {
    "classes": classes,
    "label": label,
    "human": human,
    "proxy": proxy,
    "predicates": predicates,
    "lift": lift,
    "ceiling": ceiling,
    "cascade": cascade,
    "diffuse": diffuse,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2 or args[0] not in DATASETS or args[1] not in STEPS:
        sys.exit(f"usage: run.py [{'|'.join(DATASETS)}] [{'|'.join(STEPS)}]")
    STEPS[args[1]](DATASETS[args[0]])
