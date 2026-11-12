"""Candidate datasets whose labels were given in context. Each loader returns rows and graphs."""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWBD = ROOT / "data/switchboard/switchboard-corpus/utterances.jsonl"
INTERVENE = ROOT / "data/intervene"
MELD = ROOT / "data/meld"

SWBD_ACTS = {
    "statement": "sd",
    "opinion": "sv",
    "backchannel": "b",
    "agree": "aa",
    "yes_answer": "ny",
    "no_answer": "nn",
    "yes_no_question": "qy",
    "wh_question": "qw",
    "appreciation": "ba",
    "closing": "fc",
}
SWBD_CONVERSATIONS = 250


def cliques(groups):
    pairs = set()
    for g in groups:
        g = sorted(g)
        pairs.update((g[a], g[b]) for a in range(len(g)) for b in range(a + 1, len(g)))
    return np.array(sorted(pairs))


def clean(text):
    text = re.sub(r"<[^>]*>|\{[A-Z]|[{}\[\]+/#]|--", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def switchboard():
    """Utterances of SWBD_CONVERSATIONS conversations spread over the corpus, with their dialogue acts."""
    utts = [json.loads(line) for line in open(SWBD)]
    convs = sorted({u["conversation_id"] for u in utts})
    keep = set(convs[:: max(1, len(convs) // SWBD_CONVERSATIONS)][:SWBD_CONVERSATIONS])
    utts = [u for u in utts if u["conversation_id"] in keep]
    acts = [{t.split("^")[0].split("(")[0] for _, t in u["meta"]["tag"]} for u in utts]
    rows = pd.DataFrame({"text": [clean(u["text"]) for u in utts]})
    for name, tag in SWBD_ACTS.items():
        rows[name] = [tag in a for a in acts]

    index = {u["id"]: i for i, u in enumerate(utts)}
    reply = {(min(i, index[u["reply-to"]]), max(i, index[u["reply-to"]]))
             for i, u in enumerate(utts) if u["reply-to"] in index}
    by_conv = defaultdict(list)
    for i, u in enumerate(utts):
        by_conv[u["conversation_id"]].append(i)
    conv = np.array([u["conversation_id"] for u in utts])
    speaker = np.array([u["speaker"] for u in utts])
    same = cliques(pd.Series(range(len(utts))).groupby(speaker).groups.values())
    return rows, {
        "reply": np.array(sorted(reply)),
        "conversation": cliques(by_conv.values()),
        "speaker": same[conv[same[:, 0]] != conv[same[:, 1]]],
    }


def intervene(name):
    """Posts of the Reddit or Gab hate speech conversations, a post is hateful if any segment says so."""
    d = pd.read_csv(INTERVENE / f"{name}.csv")
    index, text, hate, reply, segments = {}, [], [], set(), []
    for ids, texts, idx in zip(d.id, d.text, d.hate_speech_idx):
        ids = [line.split(". ", 1) for line in ids.strip("\n").split("\n")]
        texts = re.split(r"\n(?=\d+\. )", texts.strip("\n"))
        bad = set(json.loads(idx)) if isinstance(idx, str) else set()
        stack, segment = [], []
        for (k, pid), t in zip(ids, texts):
            depth = len(pid) - len(pid.lstrip("\t"))
            pid = pid.strip()
            if pid not in index:
                index[pid] = len(text)
                text.append(t.split(". ", 1)[1].strip() if ". " in t else "")
                hate.append(False)
            i = index[pid]
            hate[i] |= int(k) in bad
            del stack[depth:]
            if stack and stack[-1] != i:
                reply.add((min(i, stack[-1]), max(i, stack[-1])))
            stack.append(i)
            segment.append(i)
        segments.append(set(segment))
    rows = pd.DataFrame({"text": text, "hate": hate})
    return rows, {"reply": np.array(sorted(reply)), "segment": cliques(segments)}


def meld():
    """MELD utterances from Friends with the emotion annotators gave while watching the scene."""
    d = pd.concat([pd.read_csv(MELD / f"{s}_sent_emo.csv").assign(split=s) for s in ("train", "dev", "test")])
    d = d.sort_values(["split", "Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    rows = pd.DataFrame({"text": d.Utterance.str.replace("\x92", "'").str.strip()})
    for e in ("anger", "disgust", "fear", "joy", "sadness", "surprise"):
        rows[e] = (d.Emotion == e).to_numpy()

    dialogue = d.split + "/" + d.Dialogue_ID.astype(str)
    i = np.arange(1, len(d))
    reply = np.c_[i - 1, i][(dialogue.to_numpy()[i] == dialogue.to_numpy()[i - 1])]
    episode = d.Season.astype(str) + "x" + d.Episode.astype(str)
    same = cliques(pd.Series(range(len(d))).groupby(episode.to_numpy()).groups.values())
    dia = dialogue.to_numpy()
    return rows, {
        "reply": reply,
        "dialogue": cliques(pd.Series(range(len(d))).groupby(dia).groups.values()),
        "episode": same[dia[same[:, 0]] != dia[same[:, 1]]],
    }


DATASETS = {
    "meld": meld,
    "switchboard": switchboard,
    "intervene_reddit": lambda: intervene("reddit"),
    "intervene_gab": lambda: intervene("gab"),
}
