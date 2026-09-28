import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "McAuley-Lab/Amazon-Reviews-2023"
CACHE = Path(__file__).resolve().parents[2] / "data" / "hf"
FIELDS = ["parent_asin", "user_id", "rating", "title", "text"]


def load(category="All_Beauty", fields=FIELDS, limit=None):
    path = hf_hub_download(
        REPO,
        f"raw/review_categories/{category}.jsonl",
        repo_type="dataset",
        cache_dir=CACHE,
    )
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            rows.append({k: r.get(k) for k in fields})
    return pd.DataFrame(rows)
