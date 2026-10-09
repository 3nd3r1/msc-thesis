import html
import json
import re
import sqlite3
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


def stack_junk(nodes):
    """Questions with too little text to judge."""
    return pd.DataFrame({"short": nodes.text.str.len() < 200})
