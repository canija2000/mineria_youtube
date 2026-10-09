"""Calcula embeddings de texto con EmbeddingGemma 2 para los videos hidratados.

Texto por video: título + tags + inicio de la descripción (sin URLs), con el
prompt "Clustering" del modelo.

Salida: data/emb_text.parquet  (pid, youtube_id, emb[768] float32 normalizado)

Uso:
  python3 embed.py              # todos los videos vivos
  python3 embed.py --limit 200  # prueba chica
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

SRC = "data/yt8m_hydrated.parquet"
OUT = "data/emb_text.parquet"
MODEL = "google/embeddinggemma-2"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")


def build_text(row, desc_chars=600, max_tags=15):
    desc = WS_RE.sub(" ", URL_RE.sub("", row["description"] or "")).strip()[:desc_chars]
    tags = ", ".join(list(row["tags"])[:max_tags]) if row["tags"] is not None else ""
    parts = [row["title"]]
    if tags:
        parts.append(f"Tags: {tags}")
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=256)
    args = ap.parse_args()

    df = pd.read_parquet(SRC, columns=["pid", "youtube_id", "title", "description", "tags"])
    df = df[df["title"].notna()].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    texts = [build_text(r) for r in df.to_dict("records")]

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    # float16 da NaN con este modelo según el model card: usar float32 (o bfloat16).
    model = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": torch.float32})
    model.max_seq_length = args.max_seq_length
    print(f"{len(texts)} videos | device={device} | max_seq_length={model.max_seq_length}")

    t0 = time.time()
    emb = model.encode(
        texts,
        prompt_name="Clustering",
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"listo en {time.time() - t0:.0f}s | shape={emb.shape} | NaN={np.isnan(emb).any()}")

    out = df[["pid", "youtube_id"]].copy()
    out["emb"] = list(emb)
    out.to_parquet(OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
