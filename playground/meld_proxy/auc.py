"""Does the 8B proxy rank MELD utterances by emotion? AUC on a balanced sample per emotion."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT / "experiments/03_link_guided_search"))

from embed import embed
from llm import PROXY, SYSTEM, client, judge, p_yes
from run import DATASETS
from search import unit

RESULTS = Path(__file__).resolve().parent / "results"
SAMPLE = 200
FORCED = SYSTEM + "\nThe context is one line without the scene. Judge it from the line alone and always answer True or False."


def collect(path, claim, rows, nodes, system=SYSTEM):
    done = {}
    if path.exists():
        done = {d["row"]: d for d in map(json.loads, path.open())}
    api = client()

    def work(i):
        for attempt in range(8):
            try:
                v, p = judge(api, PROXY, claim, {"title": "", "text": nodes.text[i]}, system)
                return {"row": int(i), "verdict": v, "p": p}
            except Exception:
                time.sleep(2**attempt)
        return {"row": int(i), "verdict": None, "p": None}

    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(24) as pool, path.open("a") as f:
        for rec in pool.map(work, [i for i in rows if i not in done]):
            f.write(json.dumps(rec) + "\n")
    return {d["row"]: d for d in map(json.loads, path.open())}


ds = DATASETS["meld"]
nodes, _ = ds.load()
v = unit(embed(nodes.text.tolist()))
rng = np.random.default_rng(0)

lines = [f"8B proxy, plain LOTUS prompt, {SAMPLE} positives and {SAMPLE} negatives per emotion",
         "a call with no verdict scores 0, as in proxy_scores",
         "redo: the calls with no verdict asked again with a prompt that says to judge the line alone",
         "",
         f"{'emotion':>10} {'rate':>6} {'cosine':>7} {'8B':>7} {'no verdict':>11} {'redo':>7} {'no verdict':>11}"]
for key, _, claim in ds.predicates:
    y = nodes[key].to_numpy()
    rows = np.r_[rng.choice(np.where(y)[0], SAMPLE, replace=False),
                 rng.choice(np.where(~y)[0], SAMPLE, replace=False)]
    got = collect(RESULTS / f"{key}.jsonl", claim, rows, nodes)
    refused = [i for i in rows if got[int(i)]["verdict"] is None]
    redo = collect(RESULTS / "forced" / f"{key}.jsonl", claim, refused, nodes, FORCED)
    recs = [got[int(i)] for i in rows]
    both = [redo[int(i)] if r["verdict"] is None else r for i, r in zip(rows, recs)]
    s = lambda rs: [p_yes(r["verdict"], 1.0 if r["p"] is None else r["p"]) if r["verdict"] else 0.0 for r in rs]
    score, score2 = s(recs), s(both)
    cosine = v[rows] @ unit(embed([claim]))[0]
    missing = sum(r["verdict"] is None for r in recs)
    missing2 = sum(r["verdict"] is None for r in both)
    lines.append(f"{key:>10} {y.mean():>6.1%} {roc_auc_score(y[rows], cosine):>7.3f} "
                 f"{roc_auc_score(y[rows], score):>7.3f} {missing:>11} "
                 f"{roc_auc_score(y[rows], score2):>7.3f} {missing2:>11}")
    print(lines[-1], flush=True)

(RESULTS / "auc.txt").write_text("\n".join(lines) + "\n")
