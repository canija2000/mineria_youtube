"""Ingesta de videos desde el dataset de tendencias de Kaggle (canerkonuk/youtube-trending-videos-global).

El dataset (CC0) guarda cada día los videos en tendencia de muchos países, con título,
descripción, tags y categoría: alcanza para el embedding sin gastar cuota de la API de YouTube.
Se filtra por categoría (por defecto Ciencia y Tecnología + Educación) para sumar temas que el
dataset base (YT-8M) casi no tiene.

Los videos nuevos (que no están ni en la base ni en el catálogo de recibidos):
  - se embeben con EmbeddingGemma 2, igual que los que llegan por la web;
  - se guardan en `data/webapp_catalog.sqlite3` (`ReceivedStore`), así el servidor los muestra
    y `remap.py` los integra al mapa en el próximo recálculo;
  - quedan auditables en `data/kaggle_videos.parquet`.

Uso:
  kaggle datasets download canerkonuk/youtube-trending-videos-global \\
      -f youtube_trending_videos_global.parquet -p data/kaggle_trending
  python3 -m recolect.kaggle_trending data/kaggle_trending/youtube_trending_videos_global.parquet --dry-run
  python3 -m recolect.kaggle_trending data/kaggle_trending/youtube_trending_videos_global.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .received import ReceivedStore, store_path

DEFAULT_CATEGORIES = ("Science & Technology", "Education")
# El dataset trae el nombre de la categoría; la API (y _flatten) usan el número
CATEGORY_IDS = {"Film & Animation": "1", "Autos & Vehicles": "2", "Music": "10", "Pets & Animals": "15",
                "Sports": "17", "Travel & Events": "19", "Gaming": "20", "People & Blogs": "22",
                "Comedy": "23", "Entertainment": "24", "News & Politics": "25", "Howto & Style": "26",
                "Education": "27", "Science & Technology": "28", "Nonprofits & Activism": "29"}
COLUMNS = ["video_id", "video_trending__date", "video_published_at", "channel_id", "video_title",
           "video_description", "video_category_id", "video_tags", "video_duration",
           "video_view_count", "video_like_count", "video_comment_count", "channel_title"]


def _text(value) -> str:
    """Los campos vacíos llegan como None o NaN según la fila."""
    return value if isinstance(value, str) else ""


def _num(value):
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else int(number)


def load_videos(parquet_path, categories=DEFAULT_CATEGORIES) -> pd.DataFrame:
    """Una fila por video de esas categorías: la de su último día en tendencia (estadísticas más recientes)."""
    df = pd.read_parquet(parquet_path, columns=COLUMNS,
                         filters=[("video_category_id", "in", list(categories))])
    # unas pocas filas del dataset vienen con las columnas corridas (29 de 12,7 millones en nov-2026)
    df = df[df["video_id"].str.fullmatch(r"[A-Za-z0-9_-]{11}", na=False)
            & df["video_trending__date"].str.fullmatch(r"\d{4}\.\d{2}\.\d{2}", na=False)]
    return (df.sort_values("video_trending__date")
              .drop_duplicates("video_id", keep="last")
              .reset_index(drop=True))


def to_metadata(row, fetched_at: str) -> dict:
    """Misma forma que `recolect.youtube._flatten`, para que el resto del pipeline no distinga el origen."""
    tags = [t.strip() for t in _text(row.video_tags).split(",") if t.strip()]
    return {
        "youtube_id": row.video_id,
        "title": row.video_title,
        # build_text usa 600 caracteres: guardar la descripción entera solo agranda el catálogo
        # (que en Cloud Run se copia completo al bucket con cada video nuevo)
        "description": _text(row.video_description)[:1000],
        "tags": tags,
        "category_id": CATEGORY_IDS.get(row.video_category_id),
        "channel_id": row.channel_id,
        "channel_title": row.channel_title,
        "published_at": row.video_published_at,
        "thumbnail_url": f"https://i.ytimg.com/vi/{row.video_id}/hqdefault.jpg",
        "view_count": _num(row.video_view_count),
        "like_count": _num(row.video_like_count),
        "comment_count": _num(row.video_comment_count),
        "duration": row.video_duration,
        "fetched_at": fetched_at,
        "source": "kaggle:canerkonuk/youtube-trending-videos-global",
        "trending_date": row.video_trending__date,
    }


def ingest(parquet_path, data_dir="data", categories=DEFAULT_CATEGORIES, limit: int | None = None,
           dry_run: bool = False) -> dict:
    data_dir = Path(data_dir)
    store = ReceivedStore(store_path(data_dir))
    seen = {r["youtube_id"] for r in store.all()}
    hydrated = data_dir / "yt8m_hydrated.parquet"
    if hydrated.exists():
        seen.update(pd.read_parquet(hydrated, columns=["youtube_id"])["youtube_id"].astype(str))

    videos = load_videos(parquet_path, categories)
    by_category = videos["video_category_id"].value_counts().to_dict()
    new = videos[~videos["video_id"].isin(seen) & videos["video_title"].fillna("").str.strip().ne("")]
    if limit:
        new = new.tail(limit)   # los más recientes en tendencia
    print(f"[kaggle] {len(videos):,} videos únicos en {list(categories)} {by_category}; "
          f"{len(new):,} nuevos para el catálogo", flush=True)
    if dry_run or new.empty:
        return {"unique": len(videos), "new": len(new), "added": 0}

    from .embedding import TextEmbedder, build_text

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [to_metadata(row, fetched_at) for row in new.itertuples(index=False)]
    print(f"[kaggle] calculando embeddings EmbeddingGemma 2 para {len(rows):,} videos...", flush=True)
    vectors = TextEmbedder(max_seq_length=256).encode([build_text(r) for r in rows], batch_size=32)
    added = store.add_many([(r["youtube_id"], r, v) for r, v in zip(rows, vectors)])

    audit = data_dir / "kaggle_videos.parquet"
    table = pd.DataFrame(rows)
    if audit.exists():
        table = pd.concat([pd.read_parquet(audit), table], ignore_index=True).drop_duplicates("youtube_id", keep="last")
    table.to_parquet(audit, index=False)
    print(f"[kaggle] +{added:,} videos en {store.path} ({store.count():,} recibidos); tabla en {audit}", flush=True)
    return {"unique": len(videos), "new": len(new), "added": added}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", help="youtube_trending_videos_global.parquet descargado de Kaggle")
    parser.add_argument("--data", default="data", help="carpeta de datos (data o data/dev)")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                        help="categorías separadas por coma, con el nombre que usa el dataset")
    parser.add_argument("--limit", type=int, help="agrega solo los N más recientes en tendencia")
    parser.add_argument("--dry-run", action="store_true", help="solo cuenta, no embebe ni guarda")
    args = parser.parse_args()
    ingest(args.parquet, args.data, tuple(c.strip() for c in args.categories.split(",") if c.strip()),
           args.limit, args.dry_run)


if __name__ == "__main__":
    main()
