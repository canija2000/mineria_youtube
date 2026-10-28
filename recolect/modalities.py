"""Descarga de miniaturas y representaciones visuales y multimodales."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .embedding import TextEmbedder, build_text

VOCAB_URL = "https://research.google.com/youtube8m/csv/2/vocabulary.csv"


def download_thumbnails(videos: pd.DataFrame, output_dir="data/thumbs", workers=16):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ids = videos["youtube_id"].dropna().astype(str).unique().tolist()

    def fetch(video_id):
        path = destination / f"{video_id}.jpg"
        if path.exists() and path.stat().st_size > 1000:
            return video_id, str(path), True
        try:
            response = requests.get(f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg", timeout=20)
            if response.status_code == 200 and len(response.content) > 1000:
                path.write_bytes(response.content)
                return video_id, str(path), True
        except requests.RequestException:
            pass
        return video_id, None, False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(fetch, ids))
    paths = {video_id: path for video_id, path, ok in rows if ok}
    print(f"[thumbs] {len(paths):,}/{len(ids):,} miniaturas disponibles")
    return paths


def fetch_vocabulary(path="data/yt8m_vocabulary.csv") -> pd.DataFrame:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        response = requests.get(VOCAB_URL, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
    vocab = pd.read_csv(destination).dropna(subset=["Name"]).reset_index(drop=True)
    return vocab


def embed_vocabulary(vocab: pd.DataFrame, embedder: TextEmbedder, output="data/emb_vocab.parquet"):
    vectors = embedder.encode(vocab["Name"].astype(str).tolist(), batch_size=64, prompt_name=None)
    result = vocab.copy()
    result["emb"] = list(vectors)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"[vocabulary] {len(result):,} etiquetas -> {output}")


def embed_images(videos: pd.DataFrame, paths: dict[str, str], embedder: TextEmbedder,
                 image_output="data/emb_image.parquet", imgtext_output="data/emb_imgtext.parquet"):
    rows = videos[videos["youtube_id"].astype(str).isin(paths)].copy().reset_index(drop=True)
    if rows.empty:
        raise RuntimeError("No se descargó ninguna miniatura; no es posible completar las modalidades visuales")

    image_paths = [paths[str(video_id)] for video_id in rows["youtube_id"]]
    image_vectors = embedder.encode([{"image": [path]} for path in image_paths], batch_size=8, prompt_name=None)
    image_result = rows[["pid", "youtube_id"]].copy()
    image_result["emb"] = list(image_vectors)
    image_result.to_parquet(image_output, index=False)
    print(f"[image] {len(image_result):,} vectores {image_vectors.shape} -> {image_output}")

    texts = [build_text(row) for row in rows.to_dict("records")]
    multimodal_inputs = [
        {"text": f"task: clustering | query: <|image|> {text}", "image": [path]}
        for text, path in zip(texts, image_paths)
    ]
    multimodal_vectors = embedder.encode(multimodal_inputs, batch_size=8, prompt_name=None)
    multimodal_result = rows[["pid", "youtube_id"]].copy()
    multimodal_result["emb"] = list(multimodal_vectors)
    multimodal_result.to_parquet(imgtext_output, index=False)
    print(f"[img+text] {len(multimodal_result):,} vectores {multimodal_vectors.shape} -> {imgtext_output}")
    return image_result, multimodal_result


def late_fusion(text_path="data/emb_text.parquet", image_path="data/emb_image.parquet",
                output="data/emb_fusion.parquet"):
    from sklearn.preprocessing import normalize

    text = pd.read_parquet(text_path, columns=["pid", "youtube_id", "emb"])
    image = pd.read_parquet(image_path, columns=["pid", "emb"])
    joined = text.merge(image, on="pid", how="inner", suffixes=("_text", "_image"))
    fused = normalize(np.stack(joined["emb_text"]) + np.stack(joined["emb_image"])).astype(np.float32)
    result = joined[["pid", "youtube_id"]].copy()
    result["emb"] = list(fused)
    result.to_parquet(output, index=False)
    print(f"[late fusion] {len(result):,} vectores -> {output}")
    return result
