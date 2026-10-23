import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


CACHE = Path(__file__).resolve().parents[2] / "data" / "embeddings"
MODEL = "all-MiniLM-L6-v2"


def embed(texts, model=MODEL):
    key = hashlib.sha256(("\n".join(texts) + model).encode()).hexdigest()[:16]
    path = CACHE / f"{key}.npy"
    if path.exists():
        return np.load(path)

    print(f"embedding {len(texts):,} texts with {model}")
    if os.environ.get("DEEPINFRA_API_KEY"):
        vectors = remote(texts, model)
    else:
        from sentence_transformers import SentenceTransformer

        vectors = SentenceTransformer(model).encode(texts, show_progress_bar=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def remote(texts, model, batch=128):
    """Same weights on DeepInfra, the vectors match the local model."""
    from llm import client

    api = client()
    parts = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    one = lambda part: [x.embedding for x in api.embeddings.create(
        model=f"sentence-transformers/{model}", input=part).data]
    with ThreadPoolExecutor(8) as pool:
        return np.array([v for part in pool.map(one, parts) for v in part], dtype=np.float32)


def kmeans(vectors, k, seed=0):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors)


def cluster(texts, k, seed=0):
    return kmeans(embed(texts), k, seed)
