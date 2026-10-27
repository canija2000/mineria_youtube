"""Baselines de texto TF-IDF+SVD y MiniLM sobre el texto de la misma colección."""

import numpy as np
import pandas as pd

from .embedding import build_text, select_device

MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def embed_tfidf(texts: list[str], dimensions=768):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    matrix = TfidfVectorizer(max_features=50_000, min_df=2, sublinear_tf=True).fit_transform(texts)
    dimensions = min(dimensions, matrix.shape[1] - 1, matrix.shape[0] - 1)
    if dimensions < 1:
        raise ValueError("No hay suficiente texto para calcular TF-IDF + SVD")
    vectors = TruncatedSVD(n_components=dimensions, random_state=42).fit_transform(matrix)
    return normalize(vectors).astype(np.float32)


def embed_minilm(texts: list[str], batch_size=64):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM_MODEL, device=select_device())
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float32)


def build_text_baselines(source="data/yt8m_hydrated.parquet", limit=None, output_dir="data"):
    frame = pd.read_parquet(source, columns=["pid", "youtube_id", "title", "description", "tags"])
    frame = frame[frame["title"].notna()].reset_index(drop=True)
    if limit:
        frame = frame.head(limit)
    texts = [build_text(row) for row in frame.to_dict("records")]
    ids = frame[["pid", "youtube_id"]]
    for name, function in (("tfidf", embed_tfidf), ("minilm", embed_minilm)):
        vectors = function(texts)
        result = ids.copy()
        result["emb"] = list(vectors)
        output = f"{output_dir}/emb_{name}.parquet"
        result.to_parquet(output, index=False)
        print(f"[{name}] {vectors.shape} -> {output}")
