import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from llm import ORACLE, client, judge, p_yes

from data import load


RESULTS = Path(__file__).resolve().parent / "results"
BUCKETS = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, None)]
THRESHOLDS = [2, 5, 10, 20, 50]

SEED = 0
SAMPLE_BUCKETS = [(5, 9), (10, 49), (50, None)]
PRODUCTS_PER_BUCKET = 100
ROWS_PER_PRODUCT = 20

WORKERS = 8
RETRIES = 8

SHUFFLES = 5
EMBEDDER = "all-MiniLM-L6-v2"
PROPAGATE_K = 3
PROPAGATE_MIN = 5
PREDICATES = [
    "reports skin irritation or an allergic reaction",
    "says the product doesn't work as advertised",
    "suspects the product is fake",
    "mentions having sensitive skin",
    "bought it as a gift",
    "mentions another person (partner, child, friend)",
    "the review is positive",
]


def report(name):
    lines = []

    def out(line=""):
        print(line)
        lines.append(line)

    def save():
        path = RESULTS / f"{name}.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {path}")

    return out, save


def bucket(lo, hi):
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def select(sizes, lo, hi):
    return sizes >= lo if hi is None else (sizes >= lo) & (sizes <= hi)


def group_report(df, key, out):
    sizes = df.groupby(key).size()
    n_groups = len(sizes)
    n_rows = int(sizes.sum())

    out(f"\n{key}: {n_groups:,} groups, {n_rows:,} rows")
    out(f"  mean {sizes.mean():.2f}  median {sizes.median():.0f}  max {sizes.max():,}")

    out(f"\n  {'size':>8}  {'groups':>10} {'%groups':>8}  {'rows':>10} {'%rows':>7}")
    for lo, hi in BUCKETS:
        sel = select(sizes, lo, hi)
        g = int(sel.sum())
        r = int(sizes[sel].sum())
        out(
            f"  {bucket(lo, hi):>8}  {g:>10,} {g / n_groups:>7.1%}  {r:>10,} {r / n_rows:>6.1%}"
        )

    out(
        f"\n  {'at least':>8}  {'groups':>10} {'%groups':>8}  {'rows':>10} {'%rows':>7}"
    )
    for t in THRESHOLDS:
        sel = sizes >= t
        g = int(sel.sum())
        r = int(sizes[sel].sum())
        out(f"  {t:>8}  {g:>10,} {g / n_groups:>7.1%}  {r:>10,} {r / n_rows:>6.1%}")


def sizes(category="All_Beauty"):
    df = load(category)
    out, save = report("sizes")
    out(f"{category}: entity size distribution")
    group_report(df, "parent_asin", out)
    group_report(df, "user_id", out)
    save()


def sample(category="All_Beauty"):
    df = load(category)
    counts = df.groupby("parent_asin").size()
    out, save = report("sample")
    out(f"{category}: sample of {PRODUCTS_PER_BUCKET} products per bucket,")
    out(f"up to {ROWS_PER_PRODUCT} reviews each, seed {SEED}")

    parts = []
    for lo, hi in SAMPLE_BUCKETS:
        pool = counts[select(counts, lo, hi)].index.to_series()
        picked = pool.sample(min(PRODUCTS_PER_BUCKET, len(pool)), random_state=SEED)
        rows = df[df.parent_asin.isin(picked)]
        rows = rows.sample(frac=1, random_state=SEED)
        rows = rows.groupby("parent_asin").head(ROWS_PER_PRODUCT)
        parts.append(rows.assign(bucket=bucket(lo, hi)))

    out(f"\n  {'bucket':>8}  {'products':>9} {'rows':>7} {'rows/product':>13}")
    for part in parts:
        n_products = part.parent_asin.nunique()
        out(
            f"  {part.bucket.iloc[0]:>8}  {n_products:>9,} {len(part):>7,}"
            f" {len(part) / n_products:>13.1f}"
        )

    df_sample = pd.concat(parts)
    users = df_sample.groupby("user_id").size()
    out(
        f"\ntotal {len(df_sample):,} reviews, {df_sample.parent_asin.nunique():,} products"
    )
    out(
        f"reviewers with 2+ reviews in the sample: {int((users >= 2).sum()):,} of {len(users):,}"
    )

    path = RESULTS / "sample.jsonl"
    df_sample.to_json(path, orient="records", lines=True)
    save()
    print(f"wrote {path}")


def label():
    rows = pd.read_json(RESULTS / "sample.jsonl", lines=True)
    path = RESULTS / "labels.jsonl"

    done = set()
    if path.exists():
        for line in path.open():
            d = json.loads(line)
            done.add((d["row"], d["predicate"]))

    jobs = [(i, p) for i in rows.index for p in PREDICATES if (i, p) not in done]
    print(
        f"{len(rows):,} rows, {len(PREDICATES)} predicates, {len(jobs):,} calls to make"
    )
    if done:
        print(f"resuming, {len(done):,} already in {path.name}")

    api = client()

    def work(job):
        i, predicate = job
        row = rows.loc[i]
        verdict = p = None
        # 429 model-busy is routine under load, so retry rather than kill the run.
        for attempt in range(RETRIES):
            try:
                verdict, p = judge(
                    api, ORACLE, predicate, {"title": row.title, "text": row.text}
                )
                break
            except Exception:
                time.sleep(2**attempt)
        return {"row": int(i), "predicate": predicate, "verdict": verdict, "p": p}

    failed = 0
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

    out, save = report("labels")
    labels = pd.read_json(path, lines=True)
    labels["p_yes"] = [
        p_yes(v, p) if pd.notna(p) else None for v, p in zip(labels.verdict, labels.p)
    ]
    scored = labels[labels.p.notna()]
    out(f"{ORACLE}, {len(labels):,} labels over {labels.row.nunique():,} reviews")

    out(f"\n  {'yes':>6} {'mean p':>7} {'p<.5':>6}  predicate")
    for predicate in PREDICATES:
        sub = labels[labels.predicate == predicate]
        yes = (sub.verdict == "yes").mean() if len(sub) else float("nan")
        unsure = (sub.p < 0.5).mean() if sub.p.notna().any() else float("nan")
        out(f"  {yes:>6.1%} {sub.p.mean():>7.3f} {unsure:>6.1%}  {predicate}")

    # p_yes contradicts the verdict below p = 0.5, so track how often that happens.
    out(
        f"\np < 0.5 on {(scored.p < 0.5).mean():.1%} of scored labels, "
        "where p_yes is unreliable"
    )
    out(f"{1 - len(scored) / len(labels):.1%} of labels have no p, logprobs were dropped")
    save()


def pair_agreement(groups, verdicts):
    d = pd.DataFrame({"g": groups, "v": verdicts})
    c = d.groupby("g").v.agg(n="size", y=lambda s: (s == "yes").sum())
    pairs = c.n * (c.n - 1) / 2
    agree = c.y * (c.y - 1) / 2 + (c.n - c.y) * (c.n - c.y - 1) / 2
    return agree.sum() / pairs.sum() if pairs.sum() else float("nan")


def kappa(groups, verdicts, strata=None, rng=None):
    observed = pair_agreement(groups, verdicts)
    v = pd.Series(list(verdicts))
    chance = []
    for _ in range(SHUFFLES):
        if strata is None:
            shuffled = v.sample(frac=1, random_state=rng.integers(1 << 31)).to_numpy()
        else:
            shuffled = (
                v.groupby(list(strata))
                .transform(
                    lambda s: s.sample(
                        frac=1, random_state=rng.integers(1 << 31)
                    ).to_numpy()
                )
                .to_numpy()
            )
        chance.append(pair_agreement(groups, shuffled))
    chance = sum(chance) / len(chance)
    return (
        observed,
        chance,
        (observed - chance) / (1 - chance) if chance < 1 else float("nan"),
    )


def embed_clusters(rows, k):
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    text = (rows.title.fillna("") + ". " + rows.text.fillna("")).tolist()
    vectors = SentenceTransformer(EMBEDDER).encode(text, show_progress_bar=False)
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(vectors)


def propagate(df, rng):
    right = settled = calls = 0
    for _, g in df.groupby("parent_asin"):
        if len(g) < PROPAGATE_MIN:
            calls += len(g)
            continue
        seen = g.sample(PROPAGATE_K, random_state=rng.integers(1 << 31))
        rest = g.drop(seen.index)
        guess = "yes" if (seen.verdict == "yes").mean() > 0.5 else "no"
        right += (rest.verdict == guess).sum()
        settled += len(rest)
        calls += len(seen)
    return right / settled if settled else float("nan"), calls, len(df)


def compare():
    labels = pd.read_json(RESULTS / "labels.jsonl", lines=True)
    rows = pd.read_json(RESULTS / "sample.jsonl", lines=True)
    rng = np.random.default_rng(SEED)

    n_products = rows.parent_asin.nunique()
    rows["cluster"] = embed_clusters(rows, n_products)
    labels = labels.join(rows[["parent_asin", "rating", "cluster"]], on="row")

    out, save = report("compare")
    out(f"{len(labels):,} labels, {n_products:,} products, {n_products:,} clusters")
    out("\nagreement between two rows of the same group, and how far above chance")
    out(
        f"\n  {'product':>17} {'cluster':>17} {'same rating':>17}  predicate"
        f"\n  {'obs':>5} {'chance':>6} {'k':>4} {'obs':>5} {'chance':>6} {'k':>4}"
        f" {'obs':>5} {'chance':>6} {'k':>4}"
    )
    for predicate in PREDICATES:
        sub = labels[labels.predicate == predicate]
        line = ""
        for groups, strata in [
            (sub.parent_asin, None),
            (sub.cluster, None),
            (sub.parent_asin, sub.rating),
        ]:
            o, c, k = kappa(groups, sub.verdict, strata, rng)
            line += f"  {o:>5.2f} {c:>6.2f} {k:>4.2f}"
        out(f"{line}  {predicate}")

    out(
        f"\nsample {PROPAGATE_K} rows per product with {PROPAGATE_MIN}+ rows, "
        f"propagate the majority"
    )
    out(f"\n  {'accuracy':>8} {'calls':>7} {'saved':>6}  predicate")
    for predicate in PREDICATES:
        sub = labels[labels.predicate == predicate]
        accuracy, calls, total = propagate(sub, rng)
        out(f"  {accuracy:>8.1%} {calls:>7,} {1 - calls / total:>6.1%}  {predicate}")
    save()


STEPS = {"sizes": sizes, "sample": sample, "label": label, "compare": compare}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"usage: run.py [{'|'.join(STEPS)}]")
    STEPS[step]()
