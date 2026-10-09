"""Embeddings baseline sobre el MISMO texto que embed.py, para comparar contra EmbeddingGemma 2.

Métodos:
  tfidf   TF-IDF (conteo de palabras) + TruncatedSVD a 768 dims (LSA). Sin redes neuronales.
  minilm  Sentence-BERT multilingüe chico (paraphrase-multilingual-MiniLM-L12-v2, 384 dims).

Salida (mismo formato que data/emb_text.parquet: pid, youtube_id, emb normalizado):
  data/emb_tfidf.parquet
  data/emb_minilm.parquet

Uso:
  python3 baseline_model.py                    # ambos métodos
  python3 baseline_model.py --method tfidf
  python3 baseline_model.py --limit 200        # prueba chica
"""

import argparse
import time

import numpy as np
import pandas as pd

from embed import SRC, build_text  # mismo texto de entrada que EmbeddingGemma 2

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_texts(limit=None):
    df = pd.read_parquet(SRC, columns=["pid", "youtube_id", "title", "description", "tags"])
    df = df[df["title"].notna()].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df[["pid", "youtube_id"]], [build_text(r) for r in df.to_dict("records")]


def embed_tfidf(texts, dims=768):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    # min_df=2: ignora palabras que aparecen en un solo video (typos, IDs, ruido)
    # sublinear_tf: usa 1 + log(tf), así una palabra repetida 20 veces no domina
    tfidf = TfidfVectorizer(max_features=50_000, min_df=2, sublinear_tf=True)
    M = tfidf.fit_transform(texts)
    print(f"[tfidf] matriz {M.shape[0]} x {M.shape[1]} (vocabulario), densidad {M.nnz / np.prod(M.shape):.4%}")

    # TF-IDF es disperso y enorme: SVD lo comprime a un vector denso comparable con los otros
    dims = min(dims, M.shape[1] - 1, M.shape[0] - 1)
    svd = TruncatedSVD(n_components=dims, random_state=42)
    emb = svd.fit_transform(M)
    print(f"[tfidf] SVD a {dims} dims conserva {svd.explained_variance_ratio_.sum():.1%} de la varianza")
    return normalize(emb).astype(np.float32)


def embed_minilm(texts, batch_size=64):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MINILM, device=device)
    print(f"[minilm] device={device} | max_seq_length={model.max_seq_length}")
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)


METHODS = {"tfidf": embed_tfidf, "minilm": embed_minilm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=[*METHODS, "all"], default="all")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    ids, texts = load_texts(args.limit)
    for name in METHODS if args.method == "all" else [args.method]:
        t0 = time.time()
        emb = METHODS[name](texts)
        out = ids.copy()
        out["emb"] = list(emb)
        path = f"data/emb_{name}.parquet"
        out.to_parquet(path)
        print(f"[{name}] {emb.shape} en {time.time() - t0:.0f}s | NaN={np.isnan(emb).any()} -> {path}\n")


if __name__ == "__main__":
    main()
