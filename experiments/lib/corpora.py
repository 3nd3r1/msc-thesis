"""The datasets, predicates and prompts, and readers for their labels and proxy scores."""

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

from data import (
    cad_junk,
    cora_junk,
    load_cad,
    load_cad_author,
    load_cad_thread,
    load_meld,
    load_meld_dialogue,
    load_cora,
    load_stack,
    load_stack_author,
    load_synergy,
    stack_junk,
    synergy_junk,
    synergy_meta,
)
from embed import embed
from llm import ORACLE, SYSTEM, p_yes
from search import unit

LABELS = Path(__file__).resolve().parents[2] / "labels"

STRICT = SYSTEM + (
    "\nAnswer True only if the context states the claim explicitly, "
    "otherwise answer False."
)

# The 8B refuses 44% of CAD calls under plain LOTUS.
MODERATION = (
    "You are a content moderation classifier. "
    "The context may be offensive, and classifying it is the job.\n"
    + SYSTEM
    + "\nAlways answer True or False."
)

# The 8B refuses a fifth to a third of MELD lines under plain LOTUS, asking for the scene.
LINE_ALONE = SYSTEM + (
    "\nThe context is one line without the scene. "
    "Judge it from the line alone and always answer True or False."
)

CORA_PREDICATES = [
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

STACK_PREDICATES = [
    ("spss", "lexical", "the question mentions SPSS or Stata"),
    ("rcode", "lexical", "the question includes R code"),
    (
        "choose",
        "high rate",
        "the question asks which statistical method or test to use",
    ),
    (
        "assumptions",
        "topical",
        "the question is about whether an assumption of a statistical method is "
        "violated",
    ),
    (
        "interaction",
        "topical",
        "the question is about an interaction or moderation effect between variables",
    ),
    (
        "interpret",
        "row local",
        "the question asks how to interpret output that a statistics program produced",
    ),
    (
        "ownstudy",
        "author",
        "the question describes data that the asker collected themselves",
    ),
    ("econ", "field", "the question is about data from economics or finance"),
    (
        "survey",
        "field",
        "the question is about data from a questionnaire or survey of people",
    ),
    ("biomed", "field", "the question is about data from biology or medicine"),
    (
        "we",
        "control",
        "the question is written in the first person plural, using we or our",
    ),
]

# From playground/synergy_screen/pick.py, fixed here so the design does not move.
SYNERGY_REVIEWS = [
    "Walker_2018",
    "Leenaars_2020",
    "Muthu_2021",
    "Hall_2012",
    "Jeyaraman_2020",
    "Wassenaar_2017",
    "van_der_Valk_2021",
    "Nelson_2002",
    "Radjenovic_2013",
    "van_Dis_2020",
]

CAD_PREDICATES = [
    (
        "identity",
        "human",
        "the post is abusive toward an identity group, such as a gender, race, "
        "religion or sexuality",
    ),
    (
        "affiliation",
        "human",
        "the post is abusive toward people because of a group they chose to join, "
        "such as a political party or a community",
    ),
    (
        "person",
        "human",
        "the post is abusive toward a specific person, in the conversation or named",
    ),
    ("counterspeech", "human", "the post challenges or argues against abuse"),
    ("slur", "human", "the post uses a slur"),
]

MELD_PREDICATES = [
    (e, "human", f"the speaker of this line from a TV show dialogue expresses {e}")
    for e in ("anger", "disgust", "fear", "joy", "sadness", "surprise")
]

ORACLE_SOURCE = f"{ORACLE}, strict prompt"
SYNERGY_SOURCE = "the review's own screeners, final inclusion after full text"
CAD_SOURCE = "CAD's annotators, who saw the thread"
MELD_SOURCE = "MELD's annotators, who watched the scene"
HUMAN_SOURCES = (SYNERGY_SOURCE, CAD_SOURCE, MELD_SOURCE)

Dataset = namedtuple(
    "Dataset",
    "name noun load predicates junk labels source",
    defaults=(ORACLE_SOURCE,),
)


def synergy(review):
    criteria = synergy_meta()[review]["publication"]["eligibility_criteria"].strip()
    claim = "the paper meets these eligibility criteria of a systematic review\n" + criteria
    return Dataset(
        f"synergy/{review}",
        "papers",
        lambda: load_synergy(review),
        [("included", "human", claim)],
        synergy_junk,
        f"synergy/{review}",
        SYNERGY_SOURCE,
    )


def cad(name, load):
    return Dataset(name, "entries", load, CAD_PREDICATES, cad_junk, "cad", CAD_SOURCE)


def meld(name, load):
    return Dataset(name, "utterances", load, MELD_PREDICATES, cad_junk, "meld", MELD_SOURCE)


DATASETS = {
    "cora": Dataset("cora", "papers", load_cora, CORA_PREDICATES, cora_junk, "cora"),
    "stack": Dataset(
        "stack", "questions", load_stack, STACK_PREDICATES, stack_junk, "stack"
    ),
    "stack_author": Dataset(
        "stack_author",
        "questions",
        load_stack_author,
        STACK_PREDICATES,
        stack_junk,
        "stack",
    ),
    "cad": cad("cad", load_cad),
    "cad_thread": cad("cad_thread", load_cad_thread),
    "cad_author": cad("cad_author", load_cad_author),
    "meld": meld("meld", load_meld),
    "meld_dialogue": meld("meld_dialogue", load_meld_dialogue),
} | {f"synergy/{r}": synergy(r) for r in SYNERGY_REVIEWS}


def correlation(mask, neighbours):
    """Of the edges leaving a positive, the share landing on one, and the lift on it."""
    pos = np.where(mask)[0]
    degree = sum(len(neighbours[i]) for i in pos)
    hits = sum(int(mask[neighbours[i]].sum()) for i in pos)
    edge = hits / degree if degree else 0.0
    return {
        "edge": edge,
        "lift": edge / mask.mean(),
        "nbr+": sum(1 for i in pos if mask[neighbours[i]].any()) / len(pos),
        "alone": sum(1 for i in pos if len(neighbours[i]) == 0) / len(pos),
    }


def subgraph(keep, edges):
    """Induced subgraph on `keep`, reindexed. An all true mask is the identity."""
    index = -np.ones(len(keep), int)
    index[keep] = np.arange(keep.sum())
    inside = edges[keep[edges[:, 0]] & keep[edges[:, 1]]]
    return index[inside]


def vectors_for(nodes):
    return unit(embed((nodes.title + ". " + nodes.text).tolist()))


def proxy_scores(ds, n):
    """p(yes) per predicate from the proxy, 0 where the proxy has no row."""
    path = LABELS / ds.labels / "proxy.jsonl"
    if not path.exists():
        sys.exit(f"no {path}, run: run.py {ds.name} proxy")
    scored = pd.read_json(path, lines=True)
    out = {}
    for key, _, _ in ds.predicates:
        sub = scored[scored.predicate == key]
        score = np.zeros(n)
        score[sub.row.to_numpy()] = [
            p_yes(v, 1.0 if pd.isna(q) else q) for v, q in zip(sub.verdict, sub.p)
        ]
        out[key] = score
    return out


def loaded(ds):
    """The oracle labels as a mask per predicate, and the two row pools."""
    if not ds.predicates:
        sys.exit(f"no predicates for {ds.name} yet")

    nodes, edges = ds.load()
    path = LABELS / ds.labels / "labels.jsonl"
    if not path.exists():
        sys.exit(f"no {path}, run: run.py {ds.name} label")

    labels = pd.read_json(path, lines=True)
    yes = labels[labels.verdict == "yes"]
    masks = {}
    for key, _, _ in ds.predicates:
        mask = np.zeros(len(nodes), bool)
        mask[yes.row[yes.predicate == key].to_numpy()] = True
        masks[key] = mask

    junk = ds.junk(nodes).any(axis=1).to_numpy()
    repeat = (nodes.title + "\n" + nodes.text).duplicated().to_numpy()
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
