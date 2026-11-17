import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from corpora import DATASETS, MODERATION, loaded
from data import cad_rows, meld_rows
from llm import ORACLE, SERVED, SYSTEM, VERDICTS, client
from report import report

RESULTS = Path(__file__).resolve().parent / "results"
CALLS = RESULTS / "calls.jsonl"
PER_CLASS = 30
WORKERS = 24
RETRIES = 8

PROMPTS = {
    "cad": (MODERATION, "[Parent post]", "[Post]"),
    "meld": (SYSTEM + "\nAlways answer True or False.", "[Line before]", "[Line]"),
}


def neighbours(name):
    """Index of each row's neighbour along the link, -1 where it has none."""
    if name == "cad":
        rows, _ = cad_rows()
        index = {k: i for i, k in enumerate(rows.info_id)}
        return np.array([index.get(p, -1) for p in rows.parent])
    dialogue, _ = meld_rows()
    prev = np.arange(len(dialogue)) - 1
    prev[0] = -1
    prev[1:][dialogue[1:] != dialogue[:-1]] = -1
    return prev


def sample():
    """Rows to ask, per dataset and predicate, as (dataset, predicate, row, neighbour, human)."""
    picks = []
    for name in PROMPTS:
        ds = DATASETS[name]
        nodes, _, _, masks, pools = loaded(ds)
        nbr = neighbours(name)
        ok = pools[1][1] & (nbr >= 0)
        ok[ok] = nodes.text.to_numpy()[nbr[ok]].astype(bool)
        rng = np.random.default_rng(0)
        for key, _, claim in ds.predicates:
            for human in (True, False):
                rows = np.where(ok & (masks[key] == human))[0]
                for i in rng.choice(rows, min(PER_CLASS, len(rows)), replace=False):
                    picks.append((name, key, claim, int(i), int(nbr[i]), human))
    return picks


def ask(api, system, claim, text, neighbour, labels):
    above, row = labels
    context = (f"{above}: «{neighbour}»\n" if neighbour is not None else "") + f"{row}: «{text}»"
    reply = api.chat.completions.create(
        model=ORACLE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nClaim: {claim}"},
        ],
        max_tokens=8,
        temperature=0,
        logprobs=True,
    )
    SERVED.add(reply.model)
    choice = reply.choices[0]
    for t in choice.logprobs.content if choice.logprobs else []:
        word = t.token.strip().lower().strip(".")
        if word in VERDICTS:
            return VERDICTS[word], math.exp(t.logprob)
    content = (choice.message.content or "").lower()
    for word, verdict in VERDICTS.items():
        if word in content:
            return verdict, None
    return None, None


def label():
    texts = {name: DATASETS[name].load()[0].text.to_numpy() for name in PROMPTS}
    done = set()
    if CALLS.exists():
        done = {(d["dataset"], d["predicate"], d["row"], d["mode"]) for d in map(json.loads, CALLS.open())}
    jobs = [
        (*pick, mode)
        for pick in sample()
        for mode in ("alone", "context")
        if (pick[0], pick[1], pick[3], mode) not in done
    ]
    print(f"{len(jobs):,} calls to make, {len(done):,} done")
    api = client()

    def work(job):
        name, key, claim, i, j, human, mode = job
        system, *labels = PROMPTS[name]
        neighbour = texts[name][j] if mode == "context" else None
        verdict = p = None
        for attempt in range(RETRIES):
            try:
                verdict, p = ask(api, system, claim, texts[name][i], neighbour, labels)
                break
            except Exception:
                time.sleep(2**attempt)
        return {"dataset": name, "predicate": key, "row": i, "neighbour": j, "human": human, "mode": mode, "verdict": verdict, "p": p}

    failed = 0
    RESULTS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(WORKERS) as pool, CALLS.open("a", buffering=1) as f:
        for n, rec in enumerate(pool.map(work, jobs), 1):
            if rec["verdict"] is None:
                failed += 1
            else:
                f.write(json.dumps(rec) + "\n")
            if n % 200 == 0:
                print(f"  {n:,}/{len(jobs):,}")
    print(f"{failed:,} calls returned no verdict, served by {', '.join(sorted(SERVED))}")


def summary():
    calls = pd.read_json(CALLS, lines=True)
    wide = calls.pivot_table(index=["dataset", "predicate", "row", "neighbour", "human"], columns="mode", values="verdict", aggfunc="first").dropna().reset_index()
    wide["row_yes"] = wide["alone"] == "yes"
    wide["context_yes"] = wide["context"] == "yes"
    masks = {name: loaded(DATASETS[name])[3] for name in PROMPTS}
    wide["nbr_pos"] = [bool(masks[d][k][j]) for d, k, j in zip(wide.dataset, wide.predicate, wide.neighbour)]
    return wide


def results():
    wide = summary()
    out, save = report(RESULTS, "recall")
    out(f"{ORACLE}, against the human labels, on a random sample of rows with a neighbour along the link")
    out("recall and fpr from the row alone and with the neighbour's text above it")
    out("nbr+ missed and nbr+ found: share of human positives whose neighbour is a positive, among those the row alone misses and finds")
    out()
    out(f"{'dataset':>8} {'predicate':>13} {'pos':>4} {'neg':>4} {'rec row':>8} {'rec ctx':>8} {'fpr row':>8} {'fpr ctx':>8} {'nbr+ missed':>12} {'nbr+ found':>11}")
    rows = []
    for (name, key), g in wide.groupby(["dataset", "predicate"], sort=False):
        pos, neg = g[g.human], g[~g.human]
        r = {
            "rec_row": pos.row_yes.mean(),
            "rec_ctx": pos.context_yes.mean(),
            "fpr_row": neg.row_yes.mean(),
            "fpr_ctx": neg.context_yes.mean(),
        }
        rows.append(r)
        missed, found = pos[~pos.row_yes], pos[pos.row_yes]
        share = lambda d: f"{d.nbr_pos.mean():.0%}" if len(d) else "-"
        out(f"{name:>8} {key:>13} {len(pos):>4} {len(neg):>4} {r['rec_row']:>8.0%} {r['rec_ctx']:>8.0%} {r['fpr_row']:>8.0%} {r['fpr_ctx']:>8.0%} {share(missed):>12} {share(found):>11}")
    df = pd.DataFrame(rows)
    rec = 100 * (df.rec_ctx - df.rec_row).mean()
    fpr = 100 * (df.fpr_ctx - df.fpr_row).mean()
    ok = rec >= 10 and fpr <= 5
    out()
    out(f"mean recall gain {rec:+.1f} points, mean fpr gain {fpr:+.1f} points")
    out(f"fixed to pass at +10 or more recall and +5 or less fpr: {'pass' if ok else 'fail'}")
    save()

    texts = {name: DATASETS[name].load()[0].text.to_numpy() for name in PROMPTS}
    out, save = report(RESULTS, "cases")
    out("Every human positive where the row alone and the context disagree, or both miss.")
    out("alone and ctx are the 70B's verdicts, nbr+ is whether the neighbour is a human positive too.")
    pos = wide[wide.human & ~(wide.row_yes & wide.context_yes)]
    for (name, key), g in pos.groupby(["dataset", "predicate"], sort=False):
        out(f"\n## {name} {key}")
        for r in g.itertuples():
            out(f"\nrow {r.row}  alone {r.alone}  ctx {r.context}  nbr+ {'yes' if r.nbr_pos else 'no'}")
            out(f"  neighbour: {texts[name][r.neighbour][:400]}")
            out(f"  row:       {texts[name][r.row][:400]}")
    save()


if __name__ == "__main__":
    {"label": label, "report": results}[sys.argv[1]]()
