"""Screen every SYNERGY review. One pool per _ids.csv, abstracts only."""

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/synergy"
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT / "experiments/03_link_guided_search"))

from embed import embed
from run import correlation
from search import adjacency, unit

REPO = DATA / "repo/datasets"
CACHE = DATA / "cache"
MAIL = "ender.spam.guru@gmail.com"
CAP = 20
SEEDS = 20
MAX_PAIRS = 4_000_000


def chunk(ids):
    q = urllib.parse.urlencode({
        "filter": "openalex_id:" + "|".join(ids),
        "select": "id,title,abstract_inverted_index,referenced_works",
        "per-page": 50, "mailto": MAIL})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                    f"https://api.openalex.org/works?{q}", timeout=60) as r:
                return json.load(r)["results"]
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []


def fetch(ids, workers=8):
    out = {}
    parts = [ids[i:i + 50] for i in range(0, len(ids), 50)]
    with ThreadPoolExecutor(workers) as pool:
        for n, works in enumerate(pool.map(chunk, parts), 1):
            for w in works:
                out[w["id"].rsplit("/", 1)[-1]] = w
            if n % 40 == 0:
                print(f"    {min(n * 50, len(ids)):,}/{len(ids):,}", flush=True)
    return out


def abstract(w):
    inv = w.get("abstract_inverted_index")
    if not inv:
        return ""
    pos = {p: word for word, places in inv.items() for p in places}
    return " ".join(pos[k] for k in sorted(pos))


def review(path):
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"{path.stem}.json"
    if out.exists():
        return json.loads(out.read_text())
    rows = [r for r in csv.DictReader(open(path)) if r.get("openalex_id")]
    if not rows:
        out.write_text("[]")
        return []
    ids = [r["openalex_id"].rsplit("/", 1)[-1] for r in rows]
    print(f"  fetching {len(ids):,}", flush=True)
    works = fetch(ids)
    data = [{"title": w.get("title") or "", "abstract": abstract(w), "id": k,
             "refs": [x.rsplit("/", 1)[-1] for x in (w.get("referenced_works") or [])],
             "included": int(r["label_included"])}
            for r, k in zip(rows, ids) if (w := works.get(k))]
    out.write_text(json.dumps(data))
    return data


def knn(vectors, k, block=2048):
    """Top k by cosine, in row blocks so the full matrix is never held."""
    n = len(vectors)
    pairs = set()
    for s in range(0, n, block):
        sim = vectors[s:s + block] @ vectors.T
        for r in range(len(sim)):
            sim[r, s + r] = -1
        for r, top in enumerate(np.argpartition(-sim, k, axis=1)[:, :k]):
            i = s + r
            pairs.update((min(i, int(j)), max(i, int(j))) for j in top)
    return np.array(sorted(pairs))


def pairs_from(groups, cap=CAP):
    out = set()
    for g in groups:
        g = sorted(set(g))
        if 2 <= len(g) <= cap:
            out.update((g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g)))
            if len(out) > MAX_PAIRS:
                return None
    return np.array(sorted(out)) if out else np.empty((0, 2), int)


def lift_of(pairs, mask):
    if pairs is None or not len(pairs):
        return float("nan")
    a, b = pairs[:, 0], pairs[:, 1]
    leaving = int(mask[a].sum() + mask[b].sum())
    return (2 * int((mask[a] & mask[b]).sum()) / leaving) / mask.mean() if leaving else float("nan")


def ctrl(mask, adj):
    rng = np.random.default_rng(0)
    return float(np.mean([correlation(rng.permutation(mask), adj)["lift"]
                          for _ in range(SEEDS)]))


def line(name, n, rate, label, e, mask, cos=None):
    adj = adjacency(e, n)
    s = correlation(mask, adj)
    j = f"{len(set(map(tuple, e)) & cos) / len(set(map(tuple, e)) | cos):>6.3f}" if cos else " " * 6
    far = np.array(sorted(set(map(tuple, e)) - cos)) if cos else None
    f = f"{lift_of(far, mask):>5.1f}x" if cos else " " * 6
    print(f"{name:>34} {n:>6,} {rate:>6.1%} {label:>13} {len(e):>8,} "
          f"{s['lift']:>5.1f}x {ctrl(mask, adj):>5.1f}x {s['nbr+']:>6.1%} "
          f"{1 - s['alone']:>6.1%} {j} {f}", flush=True)


files = sorted(REPO.glob("*/*_ids.csv"), key=lambda p: p.stat().st_size)
print(f"{'review':>34} {'rows':>6} {'rate':>6} {'graph':>13} {'edges':>8} "
      f"{'lift':>6} {'ctrl':>6} {'nbr+':>6} {'reach':>6} {'jacc':>6} {'far':>6}", flush=True)

for path in files:
    data = [r for r in review(path) if len(r["abstract"]) > 50]
    name = path.stem.replace("_ids", "")
    if len(data) < 100 or not (sum(r["included"] for r in data)) >= 10:
        print(f"{name:>34} {len(data):>6,}   skipped, too small", flush=True)
        continue
    n = len(data)
    at = {r["id"]: i for i, r in enumerate(data)}
    mask = np.array([bool(r["included"]) for r in data])
    rate = mask.mean()

    cite = {(min(i, j), max(i, j)) for i, r in enumerate(data)
            for x in r["refs"] if (j := at.get(x)) is not None and j != i}
    cite = np.array(sorted(cite)) if cite else np.empty((0, 2), int)
    by_ref = defaultdict(list)
    for i, r in enumerate(data):
        for x in r["refs"]:
            by_ref[x].append(i)
    coupling = pairs_from(by_ref.values())

    vectors = unit(embed([f"{r['title']}. {r['abstract']}" for r in data]))

    for label, e in [("citation", cite), ("bib-coupling", coupling)]:
        if e is None:
            print(f"{name:>34} {n:>6,} {rate:>6.1%} {label:>13}   too dense", flush=True)
            continue
        if not len(e):
            continue
        k = max(1, round(len(e) / n))
        cos = set(map(tuple, knn(vectors, k)))
        line(name, n, rate, label, e, mask, cos)
        line(name, n, rate, f"cosine k={k}", np.array(sorted(cos)), mask)
    print(flush=True)
