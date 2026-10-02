import hashlib
from pathlib import Path

import numpy as np


CACHE = Path(__file__).resolve().parents[2] / "data" / "embeddings"
MODEL = "all-MiniLM-L6-v2"


def embed(texts, model=MODEL):
    key = hashlib.sha256(("\n".join(texts) + model).encode()).hexdigest()[:16]
    path = CACHE / f"{key}.npy"
    if path.exists():
        return np.load(path)

    from sentence_transformers import SentenceTransformer

    print(f"embedding {len(texts):,} texts with {model}")
    vectors = SentenceTransformer(model).encode(texts, show_progress_bar=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def cluster(texts, k, seed=0):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embed(texts))
