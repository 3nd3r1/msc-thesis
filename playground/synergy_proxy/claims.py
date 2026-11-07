import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "03_link_guided_search"))

from embed import embed
from llm import ORACLE, PROXY, SYSTEM, client, judge, p_yes
from run import DATASETS, LABELS
from search import unit


RESULTS = Path(__file__).resolve().parent / "results"

SHORT = {
    "Nelson_2002": "the paper compares clinical outcomes between users and nonusers of hormone replacement therapy",
    "van_der_Valk_2021": "the paper reports an association between hair glucocorticoid levels and obesity",
    "Jeyaraman_2020": "the paper is a trial of mesenchymal stem cell therapy for knee osteoarthritis",
    "Leenaars_2020": "the paper studies the efficacy of methotrexate in rheumatoid arthritis patients or animal models",
}


def collect(path, model, claim, nodes):
    done = {}
    if path.exists():
        done = {d["row"]: d for d in map(json.loads, path.open())}
    api = client()

    def work(i):
        row = nodes.loc[i]
        for attempt in range(8):
            try:
                v, p = judge(api, model, claim, {"title": row.title, "text": row.text})
                return {"row": i, "verdict": v, "p": p}
            except Exception:
                time.sleep(2**attempt)
        return {"row": i, "verdict": None, "p": None}

    todo = [i for i in range(len(nodes)) if i not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(24) as pool, path.open("a") as f:
        for rec in pool.map(work, todo):
            if rec["verdict"]:
                f.write(json.dumps(rec) + "\n")
    return scores(path, len(nodes))


def scores(path, n):
    s = np.zeros(n)
    for d in map(json.loads, path.open()):
        s[d["row"]] = p_yes(d["verdict"], 1.0 if d["p"] is None else d["p"])
    return s


lines = [f"{'review':>20} {'rate':>6} {'cosine':>7} {'8B full':>8} {'8B short':>9} {'70B full':>9}"]
for review, short in SHORT.items():
    ds = DATASETS[f"synergy/{review}"]
    nodes, _ = ds.load()
    y = nodes.included.to_numpy().astype(bool)
    full = ds.predicates[0][2]

    v = unit(embed((nodes.title + ". " + nodes.text).tolist()))
    cosine = v @ unit(embed([short]))[0]
    old = scores(LABELS / ds.labels / "proxy.jsonl", len(nodes))
    small = collect(RESULTS / "8b_short" / f"{review}.jsonl", PROXY, short, nodes)
    big = collect(RESULTS / "70b_full" / f"{review}.jsonl", ORACLE, full, nodes)

    aucs = [roc_auc_score(y, s) for s in (cosine, old, small, big)]
    lines.append(f"{review:>20} {y.mean():>6.1%}" + "".join(f" {a:>8.3f}" for a in aucs))
    print(lines[-1])

(RESULTS / "auc.txt").write_text(
    "AUC against the human labels, all rows, plain LOTUS prompt\n"
    "cosine is the short claim against each paper, no LLM call\n\n" + "\n".join(lines) + "\n"
)
