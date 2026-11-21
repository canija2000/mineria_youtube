"""Construcción de texto y embeddings de EmbeddingGemma 2."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# MAPA_MODELO permite usar una copia local del modelo (la imagen de Cloud Run la trae adentro)
MODEL_NAME = os.getenv("MAPA_MODELO", "google/embeddinggemma-2")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")


def _text_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def build_text(row: dict, desc_chars: int = 600, max_tags: int = 15) -> str:
    description = WS_RE.sub(" ", URL_RE.sub("", _text_value(row.get("description")))).strip()[:desc_chars]
    tags = row.get("tags")
    if tags is None or (not isinstance(tags, (list, tuple, np.ndarray)) and pd.isna(tags)):
        tags = []
    if isinstance(tags, str):
        tags = [tags]
    elif isinstance(tags, np.ndarray):
        tags = tags.tolist()
    parts = [_text_value(row.get("title"))]
    if tags:
        parts.append("Tags: " + ", ".join(list(tags)[:max_tags]))
    if description:
        parts.append(description)
    return "\n".join(part for part in parts if part)


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TextEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, max_seq_length: int = 256, image_tokens: int = 70):
        self.model = SentenceTransformer(
            model_name, device=select_device(), model_kwargs={"torch_dtype": torch.float32}
        )
        self.model.max_seq_length = max_seq_length
        self.image_tokens = image_tokens

    def enable_images(self):
        processor = self.model[0].processor
        processor.image_processor.max_soft_tokens = self.image_tokens
        processor.image_processor.image_seq_length = self.image_tokens
        processor.image_seq_length = self.image_tokens

    def encode(self, inputs: list, batch_size: int = 32, prompt_name: str | None = "Clustering") -> np.ndarray:
        kwargs = {"prompt_name": prompt_name} if prompt_name else {}
        vectors = self.model.encode(
            inputs, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
            **kwargs,
        ).astype(np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Gemma produjo embeddings con valores no finitos")
        return vectors


def embed_parquet(source: str | Path, output: str | Path, limit: int | None = None,
                  batch_size: int = 32, max_seq_length: int = 256,
                  embedder: TextEmbedder | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(source, columns=["pid", "youtube_id", "title", "description", "tags"])
    frame = frame[frame["title"].notna()].reset_index(drop=True)
    if limit:
        frame = frame.head(limit)
    texts = [build_text(row) for row in frame.to_dict("records")]
    embedder = embedder or TextEmbedder(max_seq_length=max_seq_length)
    started = time.time()
    vectors = embedder.encode(texts, batch_size=batch_size)
    result = frame[["pid", "youtube_id"]].copy()
    result["emb"] = list(vectors)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    print(f"{len(result):,} embeddings {vectors.shape} en {time.time() - started:.0f}s -> {destination}")
    return result
