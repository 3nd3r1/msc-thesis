import sys
from pathlib import Path

from data import load

RESULTS = Path(__file__).resolve().parent / "results"
BUCKETS = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, None)]
THRESHOLDS = [2, 5, 10, 20, 50]


def group_report(df, key, out):
    sizes = df.groupby(key).size()
    n_groups = len(sizes)
    n_rows = int(sizes.sum())

    out(f"\n{key}: {n_groups:,} groups, {n_rows:,} rows")
    out(f"  mean {sizes.mean():.2f}  median {sizes.median():.0f}  max {sizes.max():,}")

    out(f"\n  {'size':>8}  {'groups':>10} {'%groups':>8}  {'rows':>10} {'%rows':>7}")
    for lo, hi in BUCKETS:
        sel = sizes >= lo if hi is None else (sizes >= lo) & (sizes <= hi)
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")
        g = int(sel.sum())
        r = int(sizes[sel].sum())
        out(f"  {label:>8}  {g:>10,} {g / n_groups:>7.1%}  {r:>10,} {r / n_rows:>6.1%}")

    out(f"\n  {'at least':>8}  {'groups':>10} {'%groups':>8}  {'rows':>10} {'%rows':>7}")
    for t in THRESHOLDS:
        sel = sizes >= t
        g = int(sel.sum())
        r = int(sizes[sel].sum())
        out(f"  {t:>8}  {g:>10,} {g / n_groups:>7.1%}  {r:>10,} {r / n_rows:>6.1%}")


def sizes(category="All_Beauty"):
    df = load(category)
    lines = []

    def out(line):
        print(line)
        lines.append(line)

    out(f"{category}: entity size distribution")
    group_report(df, "parent_asin", out)
    group_report(df, "user_id", out)

    path = RESULTS / "sizes.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


STEPS = {"sizes": sizes}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"usage: run.py [{'|'.join(STEPS)}]")
    STEPS[step]()
