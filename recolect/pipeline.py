"""Ejecución completa: descarga -> metadatos -> embeddings -> mapa 3D.

Datos reales:  python3 recolect.py               (escribe en data/)
Modo prueba:   python3 recolect.py --limit 200   (escribe en data/dev/, nunca toca data/)

Las cachés que solo crecen y no dependen de cuántos videos se procesen
(shards de YT-8M, muestra extraída, id_map, respuestas de la API, miniaturas y vocabulario)
se comparten entre ambos modos y viven siempre en data/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baselines import build_text_baselines
from .download import download_shards
from .embedding import TextEmbedder, embed_parquet
from .extract import extract_tfrecords
from .mapping import build_map, current_info
from .modalities import download_thumbnails, embed_images, embed_vocabulary, fetch_vocabulary, late_fusion
from .modern import collect_modern_videos
from .youtube import hydrate_sample

DATA = Path("data")       # cachés compartidas y resultados reales
DEV = DATA / "dev"        # resultados del modo prueba (--limit)


def output_dir(limit=None) -> Path:
    return DEV if limit else DATA


def _stage(name, should_run, action):
    if should_run:
        print(f"\n=== {name} ===", flush=True)
        action()
    else:
        print(f"\n=== {name}: ya existe, se reutiliza ===", flush=True)


def consolidate(out: Path, limit=None):
    sample = pd.read_parquet(DATA / "yt8m_sample.parquet", columns=["pid", "labels"])
    if limit:
        sample = sample.head(limit)
    hydrated = pd.read_parquet(out / "yt8m_hydrated.parquet")
    columns = [column for column in (
        "pid", "youtube_id", "title", "description", "tags", "thumbnail_url",
        "channel_title", "category_id", "view_count", "like_count", "comment_count"
    ) if column in hydrated.columns]
    consolidated = sample.merge(hydrated[columns], on="pid", how="left", suffixes=("", "_api"))
    representations = ("tfidf", "minilm", "text", "image", "imgtext", "fusion")
    for name in representations:
        frame = pd.read_parquet(out / f"emb_{name}.parquet", columns=["pid", "emb"])
        consolidated = consolidated.merge(frame.rename(columns={"emb": f"emb_{name}"}), on="pid", how="left")
    output = out / "embeddings_consolidated.parquet"
    consolidated.to_parquet(output, index=False)
    print(f"[consolidated] {len(consolidated):,} videos, {len(representations)} espacios -> {output}")


def download_missing_thumbnails(limit=None):
    """Solo descarga las miniaturas que faltan. No calcula vectores ni reescribe ningún archivo."""
    hydrated_path = output_dir(limit) / "yt8m_hydrated.parquet"
    videos = pd.read_parquet(hydrated_path, columns=["youtube_id", "title"])
    if limit:
        videos = videos.head(limit)
    live_videos = videos[videos["title"].notna()]
    print(f"=== Descargar miniaturas faltantes de {len(live_videos):,} videos vivos ===", flush=True)
    download_thumbnails(live_videos, output_dir=DATA / "thumbs")


def run(limit=None, force=False):
    out = output_dir(limit)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Resultados en: {out}/" + ("  (modo prueba, data/ no se modifica)" if limit else ""), flush=True)

    # Solo se recalcula todo si cambió la cantidad de videos respecto de la última corrida
    # en esta misma carpeta. Si no hay registro previo, se confía en los archivos existentes
    # y cada etapa decide por sí misma si le falta algo.
    manifest_path = out / "recolect_manifest.json"
    selection_changed = False
    if manifest_path.exists():
        try:
            selection_changed = json.loads(manifest_path.read_text()).get("limit") != limit
        except (OSError, json.JSONDecodeError):
            selection_changed = True
    recompute_vectors = force or selection_changed

    shard_dir = DATA / "yt8m/video/train"
    has_shards = len(list(shard_dir.glob("*.tfrecord"))) >= 41
    _stage("1/11 Descargar YT-8M", not has_shards, lambda: download_shards(shard_dir))

    sample_path = DATA / "yt8m_sample.parquet"
    _stage("2/11 Extraer pids, etiquetas y features de YT-8M", not sample_path.exists(),
           lambda: extract_tfrecords(str(shard_dir / "*.tfrecord"), sample_path))

    hydrated_path = out / "yt8m_hydrated.parquet"
    _stage("3-4/11 Resolver pids y consultar YouTube Data API",
           not hydrated_path.exists() or selection_changed,
           lambda: hydrate_sample(sample_path, hydrated_path, limit=limit))

    videos = pd.read_parquet(hydrated_path)
    if limit:
        videos = videos.head(limit)
    live_videos = videos[videos["title"].notna()].copy()
    thumb_dir = DATA / "thumbs"
    video_ids = live_videos["youtube_id"].dropna().astype(str).unique()
    thumbnails_complete = all(
        (thumb_dir / f"{video_id}.jpg").is_file()
        and (thumb_dir / f"{video_id}.jpg").stat().st_size > 1000
        for video_id in video_ids
    )
    if recompute_vectors or not thumbnails_complete:
        print("\n=== 5/11 Descargar thumbnails ===", flush=True)
        thumbnail_paths = download_thumbnails(live_videos, output_dir=thumb_dir)
    else:
        thumbnail_paths = {video_id: str(thumb_dir / f"{video_id}.jpg")
                           for video_id in live_videos["youtube_id"].dropna().astype(str)}
        print("\n=== 5/11 Thumbnails ya descargados ===", flush=True)

    vocab_path = DATA / "yt8m_vocabulary.csv"
    vocab = fetch_vocabulary(vocab_path)
    print(f"\n=== 6/11 Vocabulario: {len(vocab):,} etiquetas ===", flush=True)

    text_path = out / "emb_text.parquet"
    baseline_paths = [out / "emb_tfidf.parquet", out / "emb_minilm.parquet"]
    image_path, imgtext_path = out / "emb_image.parquet", out / "emb_imgtext.parquet"
    vocab_emb_path, fusion_path = out / "emb_vocab.parquet", out / "emb_fusion.parquet"
    expected_image_ids = set(thumbnail_paths)
    existing_image_ids = set()
    existing_imgtext_ids = set()
    if image_path.exists():
        existing_image_ids = set(pd.read_parquet(image_path, columns=["youtube_id"])["youtube_id"].astype(str))
    if imgtext_path.exists():
        existing_imgtext_ids = set(pd.read_parquet(imgtext_path, columns=["youtube_id"])["youtube_id"].astype(str))
    needs_gemma = (recompute_vectors or not text_path.exists() or not image_path.exists()
                   or not imgtext_path.exists() or not vocab_emb_path.exists()
                   or existing_image_ids != expected_image_ids or existing_imgtext_ids != expected_image_ids)
    embedder = TextEmbedder(max_seq_length=256) if needs_gemma else None

    text_recomputed = recompute_vectors or not text_path.exists()
    _stage("7/11 EmbeddingGemma de texto", text_recomputed,
           lambda: embed_parquet(hydrated_path, text_path, limit, embedder=embedder))

    if recompute_vectors or not all(path.exists() for path in baseline_paths):
        print("\n=== 7/11 Baselines de texto: TF-IDF+SVD y MiniLM ===", flush=True)
        build_text_baselines(hydrated_path, limit=limit, output_dir=out)
    else:
        print("\n=== 7/11 Baselines de texto: ya existen ===", flush=True)

    if recompute_vectors or not vocab_emb_path.exists():
        embedder.model.max_seq_length = 256
        embed_vocabulary(vocab, embedder, vocab_emb_path)
    else:
        print("[vocabulary] embedding ya existe, se reutiliza")

    if (recompute_vectors or not image_path.exists() or not imgtext_path.exists()
            or existing_image_ids != expected_image_ids or existing_imgtext_ids != expected_image_ids):
        embedder.model.max_seq_length = 512
        embedder.enable_images()
        _image, _multimodal = embed_images(live_videos, thumbnail_paths, embedder,
                                            image_path, imgtext_path)
    else:
        print("\n=== 8-9/11 Embeddings de imagen e imagen+texto: ya existen ===")

    _stage("10/11 Fusión tardía", recompute_vectors or not fusion_path.exists()
           or not image_path.exists(),
           lambda: late_fusion(text_path, image_path, fusion_path))
    _stage("Consolidar metadatos y los seis espacios vectoriales", True, lambda: consolidate(out, limit))

    # El mapa se rehace si cambiaron los vectores de texto (los que usa el servidor) o si no existe.
    _stage("11/11 Mapa 3D (UMAP) para el servidor",
           force or text_recomputed or current_info(out) is None,
           lambda: build_map(out))

    manifest = {
        "limit": limit,
        "rows": len(pd.read_parquet(out / "embeddings_consolidated.parquet", columns=["pid"])),
        "map_version": (current_info(out) or {}).get("version"),
        "representations": {
            "tfidf": "emb_tfidf (TF-IDF + SVD)", "minilm": "emb_minilm (MiniLM)",
            "text": "emb_text (EmbeddingGemma texto)", "image": "emb_image (EmbeddingGemma imagen)",
            "imgtext": "emb_imgtext (EmbeddingGemma imagen + texto)",
            "fusion": "emb_fusion (normalización de texto + imagen)",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nRecolección terminada. Salida consolidada: {out / 'embeddings_consolidated.parquet'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int,
                        help="modo prueba: procesa solo N videos y escribe en data/dev/ (no toca data/)")
    parser.add_argument("--force", action="store_true", help="recalcula las etapas que generan embeddings")
    parser.add_argument("--solo-thumbnails", action="store_true",
                        help="solo descarga las miniaturas que faltan; no calcula ni reescribe nada más")
    parser.add_argument("--modern", action="store_true",
                        help="extra: recolecta videos modernos por países y canales sin tocar el flujo base")
    parser.add_argument("--regions", default="CL,AR,MX,ES,US",
                        help="códigos ISO de países separados por coma para --modern (default: CL,AR,MX,ES,US)")
    parser.add_argument("--channels", type=int, default=20,
                        help="cantidad de canales semilla a consultar en --modern (default: 20)")
    parser.add_argument("--per-channel", type=int, default=5,
                        help="últimos N videos por canal en --modern (default: 5)")
    args = parser.parse_args()
    if args.solo_thumbnails:
        download_missing_thumbnails(limit=args.limit)
        return
    if args.modern:
        out = output_dir(args.limit)
        regions = [r.strip() for r in args.regions.split(",") if r.strip()]
        collect_modern_videos(
            data_dir=out,
            regions=regions,
            channel_count=args.channels,
            per_channel=args.per_channel,
        )
        return
    run(limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
