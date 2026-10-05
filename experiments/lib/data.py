import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


CACHE = Path(__file__).resolve().parents[2] / "data" / "hf"
AMAZON = "McAuley-Lab/Amazon-Reviews-2023"
REVIEW_FIELDS = ["parent_asin", "user_id", "rating", "title", "text"]


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


CORA = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
GRAPHS = Path(__file__).resolve().parents[2] / "data" / "graphs"


def load_cora():
    """Papers with their class, binary word features and the citation edges."""
    folder = GRAPHS / "cora"
    if not (folder / "cora.content").exists():
        GRAPHS.mkdir(parents=True, exist_ok=True)
        print(f"downloading {CORA}")
        with urllib.request.urlopen(CORA) as response:
            body = io.BytesIO(response.read())
        tarfile.open(fileobj=body).extractall(GRAPHS, filter="data")

    content = pd.read_csv(folder / "cora.content", sep="\t", header=None, dtype={0: str})
    nodes = pd.DataFrame({"paper": content[0], "label": content.iloc[:, -1]})
    features = content.iloc[:, 1:-1].to_numpy(float)

    index = {paper: i for i, paper in enumerate(nodes["paper"])}
    cites = pd.read_csv(folder / "cora.cites", sep="\t", header=None, dtype=str)
    edges = np.array(
        [[index[a], index[b]] for a, b in cites.itertuples(index=False) if a in index and b in index]
    )
    return nodes, features, edges
