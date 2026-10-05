import json
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


CORA = "Graph-COM/Text-Attributed-Graphs"


def load_cora():
    """Papers with title, abstract, class and word features, and the citation edges."""
    import torch

    path = hf_hub_download(CORA, "cora/processed_data.pt", repo_type="dataset", cache_dir=CACHE)
    data = torch.load(path, weights_only=False)

    def strip(values, prefix):
        return [v.removeprefix(prefix).strip() for v in values]

    nodes = pd.DataFrame(
        {
            "title": strip(data.title, "Title:"),
            "abstract": strip(data.abs, "Abstract:"),
            "label": [data.label_texts[i] for i in data.y.tolist()],
        }
    )
    features = data.x.numpy()
    edges = np.unique(np.sort(data.edge_index.numpy().T, axis=1), axis=0)
    return nodes, features, edges
