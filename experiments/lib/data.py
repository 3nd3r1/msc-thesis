import csv
import html
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


CACHE = Path(__file__).resolve().parents[2] / "data" / "hf"
AMAZON = "McAuley-Lab/Amazon-Reviews-2023"
REVIEW_FIELDS = ["parent_asin", "user_id", "rating", "title", "text"]

# Every graph loader returns nodes with title and text, plus the edges as row pairs.
CORA = "Graph-COM/Text-Attributed-Graphs"
STACK = Path(__file__).resolve().parents[2] / "data" / "codebase_community.sqlite"
HTML = re.compile(r"<[^>]+>")


def hf_jsonl(repo, path, fields, limit=None):
    local = hf_hub_download(repo, path, repo_type="dataset", cache_dir=CACHE)
    rows = []
    with open(local) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            rows.append({k: r.get(k) for k in fields})
    return pd.DataFrame(rows)


def load_amazon(category="All_Beauty", fields=REVIEW_FIELDS, limit=None):
    return hf_jsonl(AMAZON, f"raw/review_categories/{category}.jsonl", fields, limit)


def cora():
    import torch

    path = hf_hub_download(
        CORA, "cora/processed_data.pt", repo_type="dataset", cache_dir=CACHE
    )
    return torch.load(path, weights_only=False)


def load_cora():
    """Papers with title, abstract and class, and the citation edges."""
    data = cora()

    def strip(values, prefix):
        return [v.removeprefix(prefix).strip() for v in values]

    nodes = pd.DataFrame(
        {
            "title": strip(data.title, "Title:"),
            "text": strip(data.abs, "Abstract:"),
            "label": [data.label_texts[i] for i in data.y.tolist()],
        }
    )
    edges = np.unique(np.sort(data.edge_index.numpy().T, axis=1), axis=0)
    return nodes, edges


def cora_features():
    """Cora's bundled word features, a bag of words picked for the 7 classes."""
    return cora().x.numpy()


def cora_junk(nodes):
    """Rows whose text is not an abstract. Keep them in the graph, report both ways."""
    return pd.DataFrame(
        {
            "no title": nodes.title.str.len() < 10,
            "no abstract": nodes.text.str.len() < 200,
            "caption": nodes.text.str.match(r"\s*(Figure|Fig\.|Table)\s*\d"),
            "references": nodes.text.str.count(r"\[\d+\]|\(\d{4}\)") >= 3,
        }
    )


def html_text(body):
    """HTML body as text. Code keeps its line breaks, everything else collapses."""
    text = html.unescape(HTML.sub(" ", body or ""))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def giant(pairs, n):
    """Mask of the largest connected component."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows, cols = np.array(pairs).T
    graph = coo_matrix((np.ones(len(pairs)), (rows, cols)), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    return label == np.bincount(label).argmax()


def cliques(groups):
    pairs = set()
    for group in groups:
        g = sorted(group)
        pairs.update((g[a], g[b]) for a in range(len(g)) for b in range(a + 1, len(g)))
    return np.array(sorted(pairs))


def load_stack():
    """Cross Validated questions in the postLinks giant component, and those links.

    The component is the sample. Picking rows at random would delete the edges.
    """
    db = sqlite3.connect(STACK)
    posts = pd.read_sql(
        "select Id, Title, Body, Tags, OwnerUserId from posts where PostTypeId = 1", db
    )
    links = pd.read_sql("select PostId, RelatedPostId from postLinks", db)
    db.close()

    row = {v: i for i, v in enumerate(posts.Id)}
    pairs = sorted(
        {
            (min(row[a], row[b]), max(row[a], row[b]))
            for a, b in links.itertuples(index=False)
            if a in row and b in row and row[a] != row[b]
        }
    )

    keep = giant(pairs, len(posts))
    index = -np.ones(len(posts), int)
    index[keep] = np.arange(keep.sum())
    edges = np.array([(index[a], index[b]) for a, b in pairs if keep[a] and keep[b]])

    inside = posts[keep]
    nodes = pd.DataFrame(
        {
            "title": inside.Title.fillna("").str.strip(),
            "text": inside.Body.map(html_text),
            "tags": inside.Tags.fillna("").str.replace("><", " ").str.strip("<>"),
            "owner": inside.OwnerUserId,
        }
    ).reset_index(drop=True)
    return nodes, edges


def load_stack_author():
    """The same questions, with an edge between two asked by the same person."""
    nodes, _ = load_stack()
    return nodes, cliques(nodes.groupby("owner").groups.values())


def stack_junk(nodes):
    """Questions with too little text to judge."""
    return pd.DataFrame({"short": nodes.text.str.len() < 200})


SYNERGY = Path(__file__).resolve().parents[2] / "data" / "synergy"
SYNERGY_REPO = SYNERGY / "repo"
OPENALEX = "https://api.openalex.org/works"
MAIL = "ender.spam.guru@gmail.com"


def synergy_meta():
    """Each review's entry in the SYNERGY datasets.toml, by key."""
    import tomllib

    with open(SYNERGY_REPO / "datasets.toml", "rb") as f:
        return {d["key"]: d for d in tomllib.load(f)["datasets"]}


def openalex_chunk(ids):
    query = {
        "filter": "openalex_id:" + "|".join(ids),
        "select": "id,title,abstract_inverted_index,referenced_works",
        "per-page": 50,
        "mailto": MAIL,
    }
    if key := os.environ.get("OPENALEX_API_KEY"):
        query["api_key"] = key
    url = f"{OPENALEX}?{urllib.parse.urlencode(query)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)["results"]
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OpenAlex failed on {ids[0]}, likely the daily budget")


def openalex(ids, workers=8):
    out = {}
    parts = [ids[i : i + 50] for i in range(0, len(ids), 50)]
    with ThreadPoolExecutor(workers) as pool:
        for n, works in enumerate(pool.map(openalex_chunk, parts), 1):
            for w in works:
                out[w["id"].rsplit("/", 1)[-1]] = w
            if n % 40 == 0:
                print(f"    {min(n * 50, len(ids)):,}/{len(ids):,}", flush=True)
    return out


def abstract(work):
    inverted = work.get("abstract_inverted_index")
    if not inverted:
        return ""
    at = {p: word for word, places in inverted.items() for p in places}
    return " ".join(at[k] for k in sorted(at))


def merged(data):
    """One row per work. A work some files list twice is included if either row is."""
    out = {}
    for r in data:
        if r["id"] in out:
            out[r["id"]]["included"] |= r["included"]
        else:
            out[r["id"]] = dict(r)
    return list(out.values())


def synergy_review(name):
    """Every work of one review file with its OpenAlex record, cached in data/synergy."""
    cache = SYNERGY / "cache" / f"{name}_ids.json"
    if cache.exists():
        return merged(json.loads(cache.read_text()))

    path = next((SYNERGY_REPO / "datasets").glob(f"*/{name}_ids.csv"))
    rows = [r for r in csv.DictReader(open(path)) if r.get("openalex_id")]
    ids = [r["openalex_id"].rsplit("/", 1)[-1] for r in rows]
    print(f"  fetching {len(ids):,}", flush=True)
    works = openalex(ids) if ids else {}
    data = [
        {
            "title": w.get("title") or "",
            "abstract": abstract(w),
            "id": k,
            "refs": [x.rsplit("/", 1)[-1] for x in (w.get("referenced_works") or [])],
            "included": int(r["label_included"]),
        }
        for r, k in zip(rows, ids)
        if (w := works.get(k))
    ]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return merged(data)


def synergy_pool(name):
    """The papers with an abstract, which is all the embeddings and the proxy read."""
    return [r for r in synergy_review(name) if len(r["abstract"]) > 50]


def synergy_citations(data):
    at = {r["id"]: i for i, r in enumerate(data)}
    pairs = {
        (min(i, j), max(i, j))
        for i, r in enumerate(data)
        for x in r["refs"]
        if (j := at.get(x)) is not None and j != i
    }
    return np.array(sorted(pairs)) if pairs else np.empty((0, 2), int)


def load_synergy(name):
    """One review's pool with the human inclusion label, and the citations inside it."""
    data = synergy_pool(name)
    nodes = pd.DataFrame(
        {
            "title": [r["title"] for r in data],
            "text": [r["abstract"] for r in data],
            "included": [bool(r["included"]) for r in data],
        }
    )
    return nodes, synergy_citations(data)


def synergy_junk(nodes):
    """Abstracts too short to judge, the same cut as Stack."""
    return pd.DataFrame({"short": nodes.text.str.len() < 200})


CAD = Path(__file__).resolve().parents[2] / "data" / "cad" / "raw" / "data" / "cad_v1_1.tsv"
CAD_CATEGORIES = {
    "identity": "IdentityDirectedAbuse",
    "affiliation": "AffiliationDirectedAbuse",
    "person": "PersonDirectedAbuse",
    "counterspeech": "CounterSpeech",
    "slur": "Slur",
}


def cad_rows():
    """One row per CAD entry, excluded splits dropped, with every category any annotator gave."""
    d = pd.read_csv(CAD, sep="\t", dtype=str, keep_default_na=False)
    d = d[~d.split.str.startswith("exclude")]
    rows = d.groupby("info_id", sort=True).agg(
        parent=("info_id.parent", "first"),
        thread=("info_thread.id", "first"),
        author=("meta_author", "first"),
        text=("meta_text", "first"),
        labels=("annotation_Primary", set),
    )
    rows = rows.reset_index()
    nodes = pd.DataFrame({"title": "", "text": rows.text})
    for key, category in CAD_CATEGORIES.items():
        nodes[key] = rows.labels.map(lambda s: category in s)
    return rows, nodes


def load_cad():
    """CAD entries with their category labels, and an edge from each reply to its parent."""
    rows, nodes = cad_rows()
    index = {k: i for i, k in enumerate(rows.info_id)}
    pairs = {
        (min(i, index[p]), max(i, index[p]))
        for i, p in enumerate(rows.parent)
        if p in index and index[p] != i
    }
    return nodes, np.array(sorted(pairs))


def load_cad_thread():
    """The same entries, with an edge between two in the same thread."""
    rows, nodes = cad_rows()
    return nodes, cliques(rows.groupby("thread").groups.values())


def load_cad_author():
    """The same entries, with an edge between two by the same author."""
    rows, nodes = cad_rows()
    known = rows[~rows.author.isin(["", "[deleted]"])]
    return nodes, cliques(known.groupby("author").groups.values())


def cad_junk(nodes):
    """Entries with no text."""
    return pd.DataFrame({"empty": nodes.text.str.len() == 0})


MELD = Path(__file__).resolve().parents[2] / "data" / "meld"
MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def meld_rows():
    """MELD utterances from Friends in dialogue order, all three splits, one column per emotion."""
    d = pd.concat(
        [pd.read_csv(MELD / f"{s}_sent_emo.csv").assign(split=s) for s in ("train", "dev", "test")]
    )
    d = d.sort_values(["split", "Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    nodes = pd.DataFrame({"title": "", "text": d.Utterance.str.strip()})
    for e in MELD_EMOTIONS:
        nodes[e] = (d.Emotion == e).to_numpy()
    return (d.split + "/" + d.Dialogue_ID.astype(str)).to_numpy(), nodes


def load_meld():
    """MELD utterances, with an edge from each to the one before it in the dialogue."""
    dialogue, nodes = meld_rows()
    i = np.arange(1, len(nodes))
    return nodes, np.c_[i - 1, i][dialogue[i] == dialogue[i - 1]]


def load_meld_dialogue():
    """The same utterances, with an edge between two in the same dialogue."""
    dialogue, nodes = meld_rows()
    return nodes, cliques(pd.Series(range(len(nodes))).groupby(dialogue).groups.values())
