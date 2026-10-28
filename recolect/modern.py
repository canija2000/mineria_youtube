"""Recolección complementaria de videos modernos (países + canales).

Este módulo es un extra independiente del flujo base de YT-8M:
  1. Consulta videos populares actuales por país y categoría (`videos.list(chart='mostPopular')`,
     1 unidad de cuota cada 50 videos).
  2. Consulta los últimos videos subidos por canales activos (`playlistItems.list` sobre la
     playlist `UU...` de cada canal + `videos.list` en lotes de 50).
  3. Calcula sus embeddings con EmbeddingGemma 2 y los guarda en:
     - `data/modern_videos.parquet` (tabla auditable con metadatos completos)
     - `data/webapp_catalog.sqlite3` (`ReceivedStore`), para que el servidor los proyecte
       de inmediato sobre el mapa vigente y `remap.py` los integre en el próximo recálculo.

Uso:
  python3 recolect.py --modern
  python3 recolect.py --modern --regions CL,AR,MX,ES,US --channels 25 --per-channel 5
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import numpy as np
import pandas as pd

from .embedding import TextEmbedder, build_text
from .received import ReceivedStore, store_path
from .youtube import _flatten

DEFAULT_REGIONS = ("CL", "AR", "MX", "ES", "US")
# Categorías principales de YouTube (Música, Juegos, Ciencia/Tecnología, Educación, Autos, Deportes, Entretenimiento)
CHART_CATEGORIES = (None, "10", "20", "28", "27", "2", "17", "24")


def _build_yt_client():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta YOUTUBE_API_KEY en el entorno (.env)")
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def fetch_popular_by_countries(
    yt,
    regions: tuple[str, ...] | list[str] = DEFAULT_REGIONS,
    categories: tuple[str | None, ...] = (None, "10", "20", "28", "27"),
    seen_ids: set[str] | None = None,
) -> tuple[list[dict], int]:
    """Descarga lotes de hasta 50 videos populares por país y categoría (1 unidad de cuota por llamada)."""
    seen = set(seen_ids or ())
    records: list[dict] = []
    quota_used = 0
    now = datetime.now(timezone.utc).isoformat()

    for region in regions:
        region_code = region.strip().upper()
        if not region_code:
            continue
        for cat_id in categories:
            kwargs = {
                "part": "snippet,statistics,contentDetails,topicDetails",
                "chart": "mostPopular",
                "regionCode": region_code,
                "maxResults": 50,
            }
            if cat_id is not None:
                kwargs["videoCategoryId"] = str(cat_id)
            try:
                resp = yt.videos().list(**kwargs).execute(num_retries=2)
                quota_used += 1
            except HttpError as exc:
                # Algunas categorías no tienen chart habilitado en ciertos países (400/404)
                if exc.resp.status in (400, 404):
                    continue
                if exc.resp.status == 403 and "quota" in str(exc).lower():
                    print("[modern:países] cuota diaria agotada; deteniendo consulta de países")
                    return records, quota_used
                continue

            for item in resp.get("items", []):
                yid = item.get("id")
                if not yid or yid in seen:
                    continue
                seen.add(yid)
                flat = _flatten(item, now)
                flat["harvest_source"] = f"chart:{region_code}:{cat_id or 'all'}"
                records.append(flat)

        print(f"[modern:países] {region_code}: acumulado {len(records):,} videos nuevos ({quota_used} u. cuota)", flush=True)

    return records, quota_used


def fetch_recent_from_channels(
    yt,
    channel_ids: list[str],
    per_channel: int = 5,
    seen_ids: set[str] | None = None,
) -> tuple[list[dict], int]:
    """Obtiene los últimos videos subidos por una lista de canales usando su playlist 'UU...'."""
    seen = set(seen_ids or ())
    candidate_ids: list[str] = []
    quota_used = 0

    for ch_id in channel_ids:
        ch_id = str(ch_id).strip()
        if not ch_id.startswith("UC") or len(ch_id) < 10:
            continue
        uploads_playlist = "UU" + ch_id[2:]
        try:
            resp = yt.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist,
                maxResults=min(50, max(1, per_channel)),
            ).execute(num_retries=2)
            quota_used += 1
        except HttpError as exc:
            if exc.resp.status in (400, 403, 404):
                if exc.resp.status == 403 and "quota" in str(exc).lower():
                    print("[modern:canales] cuota diaria agotada al listar playlists")
                    break
                continue
            continue

        for item in resp.get("items", []):
            vid_id = (item.get("contentDetails") or {}).get("videoId")
            if vid_id and vid_id not in seen:
                seen.add(vid_id)
                candidate_ids.append(vid_id)

    # Hidratar en lotes de 50 IDs (1 unidad de cuota cada 50 videos)
    records: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    for i in range(0, len(candidate_ids), 50):
        batch = candidate_ids[i : i + 50]
        try:
            resp = yt.videos().list(
                part="snippet,statistics,contentDetails,topicDetails",
                id=",".join(batch),
                maxResults=50,
            ).execute(num_retries=2)
            quota_used += 1
        except HttpError as exc:
            if exc.resp.status == 403 and "quota" in str(exc).lower():
                print("[modern:canales] cuota diaria agotada al hidratar lotes")
                break
            continue

        for item in resp.get("items", []):
            flat = _flatten(item, now)
            flat["harvest_source"] = "channel_uploads"
            records.append(flat)

    print(
        f"[modern:canales] {len(channel_ids)} canales consultados -> "
        f"{len(records):,} videos nuevos hidratados ({quota_used} u. cuota)",
        flush=True,
    )
    return records, quota_used


def select_seed_channels(data_dir: Path, extra_records: list[dict], max_channels: int = 25) -> list[str]:
    """Combina canales de la base hidratada (con más vistas) y canales recién descubiertos."""
    channels: list[str] = []
    seen: set[str] = set()

    # Primero canales descubiertos en los charts actuales (garantizados activos hoy)
    for rec in extra_records:
        cid = rec.get("channel_id")
        if cid and isinstance(cid, str) and cid.startswith("UC") and cid not in seen:
            seen.add(cid)
            channels.append(cid)
            if len(channels) >= max_channels // 2:
                break

    # Luego canales diversos de yt8m_hydrated.parquet
    hydrated_path = data_dir / "yt8m_hydrated.parquet"
    if hydrated_path.exists() and len(channels) < max_channels:
        df = pd.read_parquet(hydrated_path, columns=["channel_id", "view_count"]).dropna(subset=["channel_id"])
        df = df.sort_values("view_count", ascending=False).drop_duplicates("channel_id")
        for cid in df["channel_id"].astype(str):
            if cid.startswith("UC") and cid not in seen:
                seen.add(cid)
                channels.append(cid)
                if len(channels) >= max_channels:
                    break

    return channels[:max_channels]


def collect_modern_videos(
    data_dir: str | Path = "data",
    regions: tuple[str, ...] | list[str] = DEFAULT_REGIONS,
    categories: tuple[str | None, ...] = (None, "10", "20", "28", "27"),
    channel_count: int = 20,
    per_channel: int = 5,
    embedder: TextEmbedder | None = None,
) -> dict:
    """Ejecuta recolección por países + canales, embebe con Gemma 2 y guarda en SQLite y Parquet."""
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ReceivedStore(store_path(out_dir))

    # IDs ya conocidos en la base o en el almacén SQLite
    seen_ids: set[str] = {r["youtube_id"] for r in store.all()}
    hydrated_path = out_dir / "yt8m_hydrated.parquet"
    if hydrated_path.exists():
        base_ids = pd.read_parquet(hydrated_path, columns=["youtube_id"])["youtube_id"].dropna().astype(str)
        seen_ids.update(base_ids.tolist())

    yt = _build_yt_client()

    # 1. Llamado por países
    country_records, q_countries = fetch_popular_by_countries(
        yt, regions=regions, categories=categories, seen_ids=seen_ids
    )
    for r in country_records:
        seen_ids.add(r["youtube_id"])

    # 2. Llamado a canales activos
    seed_channels = select_seed_channels(out_dir, country_records, max_channels=channel_count)
    channel_records, q_channels = fetch_recent_from_channels(
        yt, seed_channels, per_channel=per_channel, seen_ids=seen_ids
    )

    all_new = country_records + channel_records
    if not all_new:
        print("[modern] No se encontraron videos nuevos adicionales en esta corrida.", flush=True)
        return {"added": 0, "from_countries": 0, "from_channels": 0, "quota_units": q_countries + q_channels}

    # 3. Calcular embeddings de texto con EmbeddingGemma 2
    embedder = embedder or TextEmbedder(max_seq_length=256)
    texts = [build_text(row) for row in all_new]
    print(f"[modern] Calculando embeddings EmbeddingGemma 2 para {len(all_new):,} videos nuevos...", flush=True)
    vectors = embedder.encode(texts, batch_size=32)

    # 4. Guardar en SQLite (ReceivedStore) para disponibilidad inmediata en servidor y remap.py
    items = [(row["youtube_id"], row, vec) for row, vec in zip(all_new, vectors)]
    inserted = store.add_many(items)

    # 5. Guardar/actualizar tabla auditable data/modern_videos.parquet
    modern_parquet = out_dir / "modern_videos.parquet"
    new_df = pd.DataFrame(all_new)
    if modern_parquet.exists():
        prev_df = pd.read_parquet(modern_parquet)
        combined = pd.concat([prev_df, new_df], ignore_index=True).drop_duplicates("youtube_id", keep="last")
    else:
        combined = new_df
    combined.to_parquet(modern_parquet, index=False)

    summary = {
        "added": inserted,
        "from_countries": len(country_records),
        "from_channels": len(channel_records),
        "quota_units": q_countries + q_channels,
        "total_received_store": store.count(),
        "parquet_path": str(modern_parquet),
    }
    print(
        f"[modern] Listo: +{inserted:,} videos guardados "
        f"(países={len(country_records)}, canales={len(channel_records)}, cuota={q_countries + q_channels} u.) "
        f"-> {modern_parquet} y {store.path}",
        flush=True,
    )
    return summary
