import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed import embed
from llm import ORACLE, PROXY, SERVED, SYSTEM, client, judge, p_yes
from report import report
from search import adjacency, search, unit

from data import cora_junk, load_cora


RESULTS = Path(__file__).resolve().parent / "results"
LABELS = Path(__file__).resolve().parents[2] / "labels" / "cora"

SEEDS = 50
RECALL = 0.90
ORDERS = ["random", "embeddings", "links", "links+embeddings"]
CASCADE_ORDERS = ["proxy", "proxy+links"]

WORKERS = 8
RETRIES = 8

STRICT = SYSTEM + (
    "\nAnswer True only if the context states the claim explicitly, "
    "otherwise answer False."
)

PREDICATES = [
    (
        "proof",
        "non-topical",
        "the abstract says the paper proves a theorem or derives a formal bound",
    ),
    ("language", "topical", "the paper is about natural language, text or speech"),
    (
        "biology",
        "topical",
        "the abstract says the method is applied to biological or medical data",
    ),
    (
        "robotics",
        "topical",
        "the paper is about robots or controlling a physical device",
    ),
    (
        "first person",
        "control",
        "the abstract is written in the first person singular, using I or my",
    ),
    (
        "markov",
        "non-topical",
        "the abstract says the paper uses a Markov model or Markov process",
    ),
    (
        "unsupervised",
        "topical",
        "the paper is about unsupervised learning or clustering",
    ),
]


def subgraph(keep, edges):
    """Induced subgraph on `keep`, reindexed. An all true mask is the identity."""
    index = -np.ones(len(keep), int)
    index[keep] = np.arange(keep.sum())
    inside = edges[keep[edges[:, 0]] & keep[edges[:, 1]]]
    return index[inside]


def table(out, names, vectors, neighbours, orders=ORDERS):
    out("  calls% +- sd" + "".join(f"{o:>22}" for o in orders))
    for name, labels, proxy in names:
        line = f"{name:>24} {labels.mean():>5.1%}"
        for order in orders:
            calls = [
                search(labels, order, vectors, neighbours, s, RECALL, proxy)
                for s in range(SEEDS)
            ]
            share = np.array(calls) / len(vectors)
            line += f"   {share.mean():>6.1%}+-{share.std():.1%}"
        out(line)


def classes():
    nodes, features, edges = load_cora()
    out, save = report(RESULTS, "classes")

    out(f"cora, {len(nodes):,} papers, {len(edges):,} citations")
    out(f"calls to {RECALL:.0%} recall, mean of {SEEDS} seeds")
    out()

    names = [
        (label, (nodes["label"] == label).to_numpy(), None)
        for label in nodes["label"].value_counts().index
    ]
    table(out, names, unit(features), adjacency(edges, len(nodes)))
    save()


def collect(model, name):
    """Ask `model` every predicate on every row, appending to labels/cora/<name>.jsonl."""
    nodes, _, _ = load_cora()
    path = LABELS / f"{name}.jsonl"

    done = set()
    if path.exists():
        for line in path.open():
            d = json.loads(line)
            done.add((d["row"], d["predicate"]))

    jobs = [
        (i, key, claim)
        for i in range(len(nodes))
        for key, _, claim in PREDICATES
        if (i, key) not in done
    ]
    print(
        f"{len(nodes):,} rows, {len(PREDICATES)} predicates, {len(jobs):,} calls to make"
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
                    {"title": row.title, "text": row.abstract},
                    system=STRICT,
                )
                break
            except Exception:
                time.sleep(2**attempt)
        return {"row": i, "predicate": key, "verdict": verdict, "p": p}

    failed = 0
    LABELS.mkdir(parents=True, exist_ok=True)
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

    out, save = report(RESULTS, name)
    labels = pd.read_json(path, lines=True)
    out(f"{model} on {labels.row.nunique():,} papers, {len(labels):,} labels")
    out("served by " + (", ".join(sorted(SERVED)) or "unknown, no calls made this run"))
    out("strict prompt, the claim has to be stated explicitly")

    out(f"\n  {'yes':>6} {'mean p':>7} {'p<.5':>6}  predicate")
    for key, _, _ in PREDICATES:
        sub = labels[labels.predicate == key]
        if len(sub):
            yes = (sub.verdict == "yes").mean()
            out(
                f"  {yes:>6.1%} {sub.p.mean():>7.3f} {(sub.p < 0.5).mean():>6.1%}  {key}"
            )

    out(f"\n{labels.p.isna().mean():.1%} of labels have no p, logprobs were dropped")
    save()


def label():
    collect(ORACLE, "labels")


def proxy():
    collect(PROXY, "proxy")


def proxy_scores(n):
    """p(yes) per predicate from the proxy, 0 where the proxy has no row."""
    path = LABELS / "proxy.jsonl"
    if not path.exists():
        sys.exit(f"no {path}, run: run.py proxy")
    scored = pd.read_json(path, lines=True)
    out = {}
    for key, _, _ in PREDICATES:
        sub = scored[scored.predicate == key]
        score = np.zeros(n)
        score[sub.row.to_numpy()] = [
            p_yes(v, 1.0 if pd.isna(q) else q)
            for v, q in zip(sub.verdict, sub.p)
        ]
        out[key] = score
    return out


def loaded():
    """The oracle labels as a mask per predicate, and the two row pools."""
    nodes, _, edges = load_cora()
    path = LABELS / "labels.jsonl"
    if not path.exists():
        sys.exit(f"no {path}, run: run.py label")

    labels = pd.read_json(path, lines=True)
    yes = labels[labels.verdict == "yes"]
    masks = {}
    for key, _, _ in PREDICATES:
        mask = np.zeros(len(nodes), bool)
        mask[yes.row[yes.predicate == key].to_numpy()] = True
        masks[key] = mask

    junk = cora_junk(nodes).any(axis=1).to_numpy()
    repeat = (nodes.title + "\n" + nodes.abstract).duplicated().to_numpy()
    keep = ~junk & ~repeat
    pools = [
        (
            f"all {len(nodes):,} rows, {int(junk.sum()):,} junk and "
            f"{int(repeat.sum())} repeated texts in the pool",
            np.ones(len(nodes), bool),
        ),
        (f"{int(keep.sum()):,} rows, junk and repeated texts dropped", keep),
    ]
    return nodes, edges, labels, masks, pools


def vectors_for(nodes):
    return unit(embed((nodes.title + ". " + nodes.abstract).tolist()))


def predicates():
    nodes, edges, labels, masks, pools = loaded()
    vectors = vectors_for(nodes)
    out, save = report(RESULTS, "predicates")

    out(f"cora, {len(nodes):,} papers, {len(edges):,} citations")
    out(f"{len(PREDICATES)} predicates labelled by {ORACLE}, strict prompt")
    out(f"calls to {RECALL:.0%} recall, mean of {SEEDS} seeds")
    out("embeddings are all-MiniLM-L6-v2 sentence vectors, not step 1's word features")

    counts = labels.groupby("predicate").size()
    gaps = len(nodes) - counts.reindex([k for k, _, _ in PREDICATES], fill_value=0)
    if gaps.any():
        short = ", ".join(f"{k} {v:,}" for k, v in gaps[gaps > 0].items())
        out(f"\nunlabelled rows, counted as no: {short}")

    for title, mask in pools:
        out(f"\n{title}")
        links = subgraph(mask, edges)
        names = [(f"{k} {kind}", masks[k][mask], None) for k, kind, _ in PREDICATES]
        table(out, names, vectors[mask], adjacency(links, int(mask.sum())))

    save()


def lift():
    """How much the citations correlate the positives. No oracle calls."""
    nodes, edges, _, masks, pools = loaded()
    out, save = report(RESULTS, "lift")

    out(f"cora, {len(nodes):,} papers, {len(edges):,} citations")
    out(f"{len(PREDICATES)} predicates labelled by {ORACLE}, strict prompt")
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
        for key, kind, _ in PREDICATES:
            m = masks[key][mask]
            pos = np.where(m)[0]
            degree = sum(len(neighbours[i]) for i in pos)
            hits = sum(int(m[neighbours[i]].sum()) for i in pos)
            any_hit = sum(1 for i in pos if m[neighbours[i]].any())
            alone = sum(1 for i in pos if len(neighbours[i]) == 0)
            rate = m.mean()
            edge = hits / degree if degree else 0.0
            out(
                f"  {rate:>6.1%} {edge:>6.1%} {edge / rate:>5.1f}x "
                f"{any_hit / len(pos):>6.1%} {alone / len(pos):>6.1%}"
                f"  {key} {kind}"
            )

    save()


def cascade():
    """Oracle calls once the proxy has scored every row, which both orders pay."""
    nodes, edges, _, masks, pools = loaded()
    scores = proxy_scores(len(nodes))
    vectors = vectors_for(nodes)
    out, save = report(RESULTS, "cascade")

    out(f"cora, {len(nodes):,} papers, {len(edges):,} citations")
    out(f"calls to {RECALL:.0%} recall, mean of {SEEDS} seeds")
    out(f"proxy is p(yes) from {PROXY}, one call per row and predicate")
    out("both orders pay that, so only the oracle calls below differ")
    out("proxy+links sends the oracle to the best proxy score among the unasked")
    out("neighbours of a confirmed positive, and to the global best when there are none")

    for title, mask in pools:
        out(f"\n{title}")
        links = subgraph(mask, edges)
        names = [
            (f"{k} {kind}", masks[k][mask], scores[k][mask])
            for k, kind, _ in PREDICATES
        ]
        table(
            out,
            names,
            vectors[mask],
            adjacency(links, int(mask.sum())),
            CASCADE_ORDERS,
        )

    save()


STEPS = {
    "classes": classes,
    "label": label,
    "proxy": proxy,
    "predicates": predicates,
    "lift": lift,
    "cascade": cascade,
}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"usage: run.py [{'|'.join(STEPS)}]")
    STEPS[step]()
