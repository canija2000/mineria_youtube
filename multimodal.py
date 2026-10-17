"""Test multimodal: embeddings de miniatura e imagen+texto con EmbeddingGemma 2 sobre una muestra.

Pasos:
  1. Muestra aleatoria de N videos vivos (seed fija).
  2. Descarga miniaturas de i.ytimg.com (mqdefault 320x180, sin bordes negros; no gasta cuota de la API).
  3. Embeddings:
       image    solo la miniatura
       imgtext  miniatura + texto en un solo input ("<|image|> " + build_text)
  4. Zero-shot tagging: ranking de las 3.862 etiquetas de YT-8M para cada miniatura.

Salidas:
  data/thumbs/{youtube_id}.jpg
  data/emb_image.parquet, data/emb_imgtext.parquet   (pid, youtube_id, emb)
  data/zeroshot_image.parquet                         (pid, top10 etiquetas predichas)
  data/emb_vocab.parquet                              (etiquetas YT-8M: Index, Name, Vertical1, emb)

Uso:
  python3 multimodal.py --n 5000
  python3 multimodal.py --n 50        # prueba chica
"""

import argparse
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer

from embed import MODEL, SRC, build_text

# La red local tiene IPv6 roto: forzar IPv4.
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **kw: _orig_getaddrinfo(
    host, port, socket.AF_INET, *a, **kw
)

THUMBS = "data/thumbs"
VOCAB_URL = "https://research.google.com/youtube8m/csv/2/vocabulary.csv"
# Tokens visuales por imagen: 280 (default) ~1,2 img/s; 70 ~6 img/s en un M2 y el embedding casi no cambia.
IMAGE_TOKENS = 70


def sample_videos(n, seed=42):
    df = pd.read_parquet(
        SRC, columns=["pid", "youtube_id", "title", "description", "tags"]
    )
    df = df[df["title"].notna()]
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def download_thumbs(youtube_ids, workers=16):
    os.makedirs(THUMBS, exist_ok=True)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers))

    def fetch(yid):
        path = f"{THUMBS}/{yid}.jpg"
        if os.path.exists(path):
            return yid, True
        try:
            r = session.get(f"https://i.ytimg.com/vi/{yid}/mqdefault.jpg", timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(path, "wb") as f:
                    f.write(r.content)
                return yid, True
        except requests.RequestException:
            pass
        return yid, False

    with ThreadPoolExecutor(workers) as ex:
        ok = dict(ex.map(fetch, youtube_ids))
    print(f"[thumbs] {sum(ok.values())}/{len(ok)} miniaturas disponibles")
    return ok


def load_model():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = SentenceTransformer(
        MODEL, device=device, model_kwargs={"torch_dtype": torch.float32}
    )
    model.max_seq_length = 512  # texto (~256) + imagen (70) caben de sobra
    proc = model[0].processor
    proc.image_processor.max_soft_tokens = IMAGE_TOKENS
    proc.image_processor.image_seq_length = IMAGE_TOKENS
    proc.image_seq_length = IMAGE_TOKENS
    return model


def encode(model, inputs, label, batch_size=8):
    t0 = time.time()
    emb = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(
        f"[{label}] {emb.shape} en {time.time() - t0:.0f}s | NaN={np.isnan(emb).any()}"
    )
    return emb


def save(df, emb, path):
    out = df[["pid", "youtube_id"]].copy()
    out["emb"] = list(emb)
    out.to_parquet(path)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    df = sample_videos(args.n)
    ok = download_thumbs(df["youtube_id"].tolist())
    df = df[df["youtube_id"].map(ok)].reset_index(drop=True)
    paths = [f"{THUMBS}/{y}.jpg" for y in df["youtube_id"]]

    model = load_model()

    # 1. solo imagen (sin prompt: los prefijos de tarea son solo para texto)
    emb_img = encode(model, [{"image": [p]} for p in paths], "image")
    save(df, emb_img, "data/emb_image.parquet")

    # 2. imagen + texto en un solo input. La imagen va primero para que, si el texto
    #    se trunca, se pierda el final de la descripción y no la imagen.
    texts = [build_text(r) for r in df.to_dict("records")]
    inputs = [
        {"text": f"task: clustering | query: <|image|> {t}", "image": [p]}
        for t, p in zip(texts, paths)
    ]
    emb_it = encode(model, inputs, "imgtext")
    save(df, emb_it, "data/emb_imgtext.parquet")

    # 3. zero-shot: qué etiquetas de YT-8M están más cerca de cada miniatura
    vocab = (
        pd.read_csv(VOCAB_URL, usecols=["Index", "Name", "Vertical1"])
        .dropna(subset=["Name"])
        .reset_index(drop=True)
    )
    lab = encode(model, vocab["Name"].tolist(), "vocab", batch_size=64)
    vocab.assign(emb=list(lab.astype(np.float16))).to_parquet("data/emb_vocab.parquet")
    print("  -> data/emb_vocab.parquet")
    top = np.argsort(-(emb_img @ lab.T), axis=1)[:, :10]
    zs = df[["pid", "youtube_id", "title"]].copy()
    zs["pred_idx"] = [vocab["Index"].to_numpy()[r].tolist() for r in top]
    zs["pred_names"] = [vocab["Name"].to_numpy()[r].tolist() for r in top]
    zs.to_parquet("data/zeroshot_image.parquet")
    print("  -> data/zeroshot_image.parquet")


if __name__ == "__main__":
    main()
