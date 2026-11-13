"""Does the 8B's refusal rate on a random sample track what links add to the proxy.

Fixed before looking:
- 300 random rows of the pool with junk and repeated texts dropped, seed 0, the same rows for every predicate of a dataset
- plain LOTUS prompt on every dataset, a refusal is a reply with no True or False in it
- gain is proxy+links minus proxy at 90% recall, as in residual_screen
- passes if the Spearman of refusal rate against gain over every predicate and edge set is -0.3 or lower
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT / "experiments/03_link_guided_search"))

from llm import PROXY, client, judge
from report import report
from run import DATASETS, loaded, proxy_scores, subgraph
from search import adjacency, search

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["cora", "stack", "stack_author", "cad", "cad_thread", "cad_author", "meld", "meld_dialogue"]
SAMPLE = 300
TARGET = 0.90
PASS = -0.3


def collect(ds, nodes, rows):
    path = RESULTS / f"{ds.labels}.jsonl"
    done = set()
    if path.exists():
        done = {(d["row"], d["predicate"]) for d in map(json.loads, path.open())}
    jobs = [(int(i), key, claim) for key, _, claim in ds.predicates for i in rows if (int(i), key) not in done]
    api = client()

    def work(job):
        i, key, claim = job
        row = {"title": nodes.title[i], "text": nodes.text[i]}
        for attempt in range(8):
            try:
                verdict, _ = judge(api, PROXY, claim, row)
                return {"row": i, "predicate": key, "verdict": verdict}
            except Exception:
                time.sleep(2**attempt)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(24) as pool, path.open("a", buffering=1) as f:
        for rec in pool.map(work, jobs):
            if rec:
                f.write(json.dumps(rec) + "\n")
    got = pd.read_json(path, lines=True)
    got = got[got.row.isin(rows)]
    return got.verdict.isna().groupby(got.predicate).mean()


refused = {}
rows = []
for name in NAMES:
    ds = DATASETS[name]
    nodes, edges, _, masks, pools = loaded(ds)
    mask = pools[1][1]
    if ds.labels not in refused:
        sample = np.random.default_rng(0).choice(np.where(mask)[0], SAMPLE, replace=False)
        refused[ds.labels] = collect(ds, nodes, sample)
    scores = proxy_scores(ds, len(nodes))
    sub = subgraph(mask, edges)
    neighbours = adjacency(sub, int(mask.sum()))
    for key, _, _ in ds.predicates:
        y = masks[key][mask].astype(float)
        p = scores[key][mask]
        calls = {o: search(y, o, None, neighbours, 0, (TARGET,), p)[0] for o in ("proxy", "proxy+links")}
        rows.append({
            "dataset": name,
            "predicate": key,
            "refused": refused[ds.labels][key],
            "gain": 100 * (calls["proxy+links"] - calls["proxy"]) / len(y),
        })
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
rho = df.refused.corr(df.gain, method="spearman")
out, save = report(RESULTS, "refusal")
out(f"refused: share of {SAMPLE} random pool rows the 8B gives no verdict on under the plain LOTUS prompt, seed 0")
out(f"gain: proxy+links minus proxy in points of the rows at {TARGET:.0%} recall, negative means links help")
out("pool with junk and repeated texts dropped")
out()
out(f"{'dataset':>13} {'predicate':>13} {'refused':>8} {'gain':>6}")
for r in df.itertuples():
    out(f"{r.dataset:>13} {r.predicate:>13} {r.refused:>8.1%} {r.gain:>+6.1f}")
out()
out(f"Spearman of refused against gain over all predicates {rho:+.3f}, fixed to pass at {PASS} or lower: {'pass' if rho <= PASS else 'fail'}")
save()
