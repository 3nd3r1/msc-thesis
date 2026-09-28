import sys
from pathlib import Path

import pandas as pd

from data import load

RESULTS = Path(__file__).resolve().parent / "results"
BUCKETS = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, None)]
THRESHOLDS = [2, 5, 10, 20, 50]

SEED = 0
SAMPLE_BUCKETS = [(5, 9), (10, 49), (50, None)]
PRODUCTS_PER_BUCKET = 100
ROWS_PER_PRODUCT = 20


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


def label(lo, hi):
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
        out(f"  {label(lo, hi):>8}  {g:>10,} {g / n_groups:>7.1%}  {r:>10,} {r / n_rows:>6.1%}")

    out(f"\n  {'at least':>8}  {'groups':>10} {'%groups':>8}  {'rows':>10} {'%rows':>7}")
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
        parts.append(rows.assign(bucket=label(lo, hi)))

    out(f"\n  {'bucket':>8}  {'products':>9} {'rows':>7} {'rows/product':>13}")
    for part in parts:
        n_products = part.parent_asin.nunique()
        out(
            f"  {part.bucket.iloc[0]:>8}  {n_products:>9,} {len(part):>7,}"
            f" {len(part) / n_products:>13.1f}"
        )

    df_sample = pd.concat(parts)
    users = df_sample.groupby("user_id").size()
    out(f"\ntotal {len(df_sample):,} reviews, {df_sample.parent_asin.nunique():,} products")
    out(f"reviewers with 2+ reviews in the sample: {int((users >= 2).sum()):,} of {len(users):,}")

    path = RESULTS / "sample.jsonl"
    df_sample.to_json(path, orient="records", lines=True)
    save()
    print(f"wrote {path}")


STEPS = {"sizes": sizes, "sample": sample}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"usage: run.py [{'|'.join(STEPS)}]")
    STEPS[step]()
