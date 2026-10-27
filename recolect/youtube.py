"""Mapea los IDs anonimizados de YT-8M a IDs reales de YouTube y los hidrata con la API.

Etapas (ambas reanudables, guardan caché en data/):
  1. resolve: pid -> youtube_id         -> data/id_map.parquet
  2. hydrate: youtube_id -> metadata API -> data/yt8m_hydrated.parquet

Uso:
  python3 -m recolect.youtube              # todo el sample
  python3 -m recolect.youtube --limit 200  # prueba chica
  (la corrida completa parte en `python3 recolect.py`)
"""

import argparse
import json
import os
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# La red local tiene IPv6 roto: forzar IPv4.
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **kw: _orig_getaddrinfo(
    host, port, socket.AF_INET, *a, **kw
)

SAMPLE = "data/yt8m_sample.parquet"
ID_MAP = "data/id_map.parquet"
RAW_API = "data/api_raw.jsonl"
OUT = "data/yt8m_hydrated.parquet"

ID_RE = re.compile(r'i\("([^"]+)","([^"]+)"\)')


# ---------- 1. pid -> youtube_id ----------

def _resolve_one(session, pid):
    url = f"http://data.yt8m.org/2/j/i/{pid[:2]}/{pid}.js"
    for attempt in range(4):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                m = ID_RE.search(r.text)
                return pid, m.group(2) if m else None
            if r.status_code in (403, 404):  # pid sin mapeo (video retirado)
                return pid, None
        except requests.RequestException:
            pass
        time.sleep(2**attempt)
    return pid, None


def resolve(pids, workers=16):
    done = pd.read_parquet(ID_MAP) if os.path.exists(ID_MAP) else pd.DataFrame(columns=["pid", "youtube_id"])
    todo = sorted(set(pids) - set(done["pid"]))
    print(f"[resolve] {len(done)} en caché, {len(todo)} por resolver")
    if not todo:
        return done

    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=workers))
    results = []
    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(lambda p: _resolve_one(session, p), todo), 1):
            results.append(res)
            if i % 2000 == 0 or i == len(todo):
                done = pd.concat([done, pd.DataFrame(results, columns=["pid", "youtube_id"])], ignore_index=True)
                done.to_parquet(ID_MAP)
                results = []
                print(f"[resolve] {i}/{len(todo)}")
    return done


# ---------- 2. youtube_id -> metadata ----------

def _flatten(item, fetched_at):
    sn, st = item.get("snippet", {}), item.get("statistics", {})
    cd, td = item.get("contentDetails", {}), item.get("topicDetails", {})
    thumbs = sn.get("thumbnails", {})
    thumb = next((thumbs[k]["url"] for k in ("maxres", "standard", "high", "medium", "default") if k in thumbs), None)
    num = lambda k: int(st[k]) if k in st else None
    return {
        "youtube_id": item["id"],
        "title": sn.get("title"),
        "description": sn.get("description"),
        "tags": sn.get("tags", []),
        "category_id": sn.get("categoryId"),
        "channel_id": sn.get("channelId"),
        "channel_title": sn.get("channelTitle"),
        "published_at": sn.get("publishedAt"),
        "default_language": sn.get("defaultLanguage"),
        "audio_language": sn.get("defaultAudioLanguage"),
        "thumbnail_url": thumb,
        "view_count": num("viewCount"),
        "like_count": num("likeCount"),
        "comment_count": num("commentCount"),
        "duration": cd.get("duration"),
        "definition": cd.get("definition"),
        "caption": cd.get("caption"),
        "topic_categories": td.get("topicCategories", []),
        "fetched_at": fetched_at,
    }


def hydrate(youtube_ids):
    seen = set()
    if os.path.exists(RAW_API):
        with open(RAW_API) as f:
            seen = {json.loads(line)["requested_id"] for line in f}
    todo = [y for y in youtube_ids if y not in seen]
    print(f"[hydrate] {len(seen)} en caché, {len(todo)} por consultar ({-(-len(todo) // 50)} unidades de cuota)")

    yt = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"], cache_discovery=False)
    with open(RAW_API, "a") as out:
        for i in range(0, len(todo), 50):
            batch = todo[i : i + 50]
            try:
                resp = yt.videos().list(
                    part="snippet,statistics,contentDetails,topicDetails", id=",".join(batch), maxResults=50
                ).execute(num_retries=3)
            except HttpError as e:
                if e.resp.status == 403 and "quota" in str(e).lower():
                    print("[hydrate] cuota diaria agotada; vuelve a correr mañana (se reanuda solo)")
                    break
                raise
            now = datetime.now(timezone.utc).isoformat()
            found = {it["id"]: it for it in resp.get("items", [])}
            for yid in batch:  # una línea por ID pedido, también los eliminados
                out.write(json.dumps({"requested_id": yid, "fetched_at": now, "item": found.get(yid)}) + "\n")
            out.flush()
            if (i // 50) % 50 == 0:
                print(f"[hydrate] {i + len(batch)}/{len(todo)}")

    rows = []
    with open(RAW_API) as f:
        for line in f:
            rec = json.loads(line)
            if rec["item"]:
                rows.append(_flatten(rec["item"], rec["fetched_at"]))
    return pd.DataFrame(rows)


def hydrate_sample(sample_path=SAMPLE, output_path=OUT, limit=None):
    load_dotenv()

    sample = pd.read_parquet(sample_path, columns=["pid", "labels"])
    if limit:
        sample = sample.head(limit)

    id_map = resolve(sample["pid"].tolist())
    id_map = id_map[id_map["pid"].isin(sample["pid"])]
    ids = id_map["youtube_id"].dropna().unique().tolist()

    meta = hydrate(ids)
    df = sample.merge(id_map, on="pid", how="left").merge(meta, on="youtube_id", how="left")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)

    n = len(df)
    mapped = df["youtube_id"].notna().sum()
    alive = df["title"].notna().sum()
    print(f"\n{n} videos | con ID real: {mapped} ({mapped / n:.1%}) | vivos en YouTube: {alive} ({alive / n:.1%})")
    print(f"-> {output_path}  (une con mean_rgb/mean_audio por 'pid')")


def fetch_video(youtube_id):
    """Consulta un video por ID para el flujo interactivo del servidor."""
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta YOUTUBE_API_KEY en el entorno")
    yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    response = yt.videos().list(
        part="snippet,statistics,contentDetails,topicDetails", id=youtube_id, maxResults=1
    ).execute(num_retries=3)
    items = response.get("items", [])
    if not items:
        return None
    return _flatten(items[0], datetime.now(timezone.utc).isoformat())


def fetch_profile_videos(query: str, max_videos: int = 40) -> tuple[dict, list[dict]] | None:
    """Resuelve un canal (@handle, URL de canal o playlist pública) y devuelve (perfil_meta, últimos_videos).

    Costo de cuota: ~3 unidades (1 channels.list + 1 playlistItems.list + 1 videos.list).
    """
    from urllib.parse import parse_qs, unquote, urlparse

    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta YOUTUBE_API_KEY en el entorno")
    yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    raw = query.strip()
    channel_id = None
    handle = None
    username = None
    playlist_id = None

    if raw.startswith("@"):
        handle = raw
    elif re.fullmatch(r"UC[A-Za-z0-9_-]{20,}", raw):
        channel_id = raw
    elif re.fullmatch(r"(?:PL|UU|LL|FL)[A-Za-z0-9_-]{10,}", raw):
        playlist_id = raw
    elif "youtube.com" in raw or "youtu.be" in raw:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        qs = parse_qs(parsed.query)
        if "list" in qs and qs["list"]:
            playlist_id = qs["list"][0]
        path = unquote(parsed.path.strip("/"))
        parts = [p for p in path.split("/") if p]
        if parts:
            if parts[0].startswith("@"):
                handle = parts[0]
            elif parts[0] == "channel" and len(parts) >= 2:
                channel_id = parts[1]
            elif parts[0] in ("user", "c") and len(parts) >= 2:
                username = parts[1]
            elif not playlist_id and not handle and not channel_id:
                handle = "@" + parts[0].lstrip("@")
    else:
        handle = "@" + raw.lstrip("@")

    profile_info = {"query": raw, "title": raw, "channel_id": None, "thumbnail_url": None, "subscriber_count": None}

    if not playlist_id:
        ch_item = None
        if channel_id:
            resp = yt.channels().list(part="snippet,contentDetails,statistics", id=channel_id, maxResults=1).execute(num_retries=2)
            ch_item = (resp.get("items") or [None])[0]
        if not ch_item and handle:
            resp = yt.channels().list(part="snippet,contentDetails,statistics", forHandle=handle, maxResults=1).execute(num_retries=2)
            ch_item = (resp.get("items") or [None])[0]
        if not ch_item and username:
            resp = yt.channels().list(part="snippet,contentDetails,statistics", forUsername=username, maxResults=1).execute(num_retries=2)
            ch_item = (resp.get("items") or [None])[0]
            if not ch_item:
                resp = yt.channels().list(part="snippet,contentDetails,statistics", forHandle="@" + username, maxResults=1).execute(num_retries=2)
                ch_item = (resp.get("items") or [None])[0]

        if not ch_item:
            return None

        sn = ch_item.get("snippet", {})
        st = ch_item.get("statistics", {})
        cd = ch_item.get("contentDetails", {})
        thumbs = sn.get("thumbnails", {})
        thumb = next((thumbs[k]["url"] for k in ("high", "medium", "default") if k in thumbs), None)
        cid = ch_item["id"]
        playlist_id = (cd.get("relatedPlaylists") or {}).get("uploads") or ("UU" + cid[2:] if cid.startswith("UC") else None)
        profile_info = {
            "query": raw,
            "channel_id": cid,
            "title": sn.get("title") or raw,
            "custom_url": sn.get("customUrl") or handle or "",
            "description": (sn.get("description") or "")[:300],
            "thumbnail_url": thumb,
            "subscriber_count": int(st["subscriberCount"]) if "subscriberCount" in st else None,
            "video_count": int(st["videoCount"]) if "videoCount" in st else None,
        }

    if not playlist_id:
        return None

    pl_resp = yt.playlistItems().list(
        part="contentDetails,snippet",
        playlistId=playlist_id,
        maxResults=min(50, max(5, max_videos)),
    ).execute(num_retries=2)
    pl_items = pl_resp.get("items", [])
    if not pl_items:
        return None

    if not profile_info.get("channel_id") and pl_items:
        first_sn = pl_items[0].get("snippet", {})
        profile_info["title"] = first_sn.get("channelTitle") or profile_info["title"]
        profile_info["channel_id"] = first_sn.get("channelId")

    video_ids = []
    seen = set()
    for it in pl_items:
        vid_id = (it.get("contentDetails") or {}).get("videoId")
        if vid_id and vid_id not in seen:
            seen.add(vid_id)
            video_ids.append(vid_id)

    if not video_ids:
        return None

    now = datetime.now(timezone.utc).isoformat()
    v_resp = yt.videos().list(
        part="snippet,statistics,contentDetails,topicDetails",
        id=",".join(video_ids[:50]),
        maxResults=50,
    ).execute(num_retries=2)
    videos = [_flatten(item, now) for item in v_resp.get("items", [])]
    if not videos:
        return None
    return profile_info, videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="procesar solo los primeros N videos")
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()
    hydrate_sample(args.sample, args.output, args.limit)


if __name__ == "__main__":
    main()
