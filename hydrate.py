"""Mapea los IDs anonimizados de YT-8M a IDs reales de YouTube y los hidrata con la API.

Etapas (ambas reanudables, guardan caché en data/):
  1. resolve: pid -> youtube_id         -> data/id_map.parquet
  2. hydrate: youtube_id -> metadata API -> data/yt8m_hydrated.parquet

Uso:
  python3 hydrate.py              # todo el sample
  python3 hydrate.py --limit 200  # prueba chica
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="procesar solo los primeros N videos")
    args = ap.parse_args()
    load_dotenv()

    sample = pd.read_parquet(SAMPLE, columns=["pid", "labels"])
    if args.limit:
        sample = sample.head(args.limit)

    id_map = resolve(sample["pid"].tolist())
    id_map = id_map[id_map["pid"].isin(sample["pid"])]
    ids = id_map["youtube_id"].dropna().unique().tolist()

    meta = hydrate(ids)
    df = sample.merge(id_map, on="pid", how="left").merge(meta, on="youtube_id", how="left")
    df.to_parquet(OUT)

    n = len(df)
    mapped = df["youtube_id"].notna().sum()
    alive = df["title"].notna().sum()
    print(f"\n{n} videos | con ID real: {mapped} ({mapped / n:.1%}) | vivos en YouTube: {alive} ({alive / n:.1%})")
    print(f"-> {OUT}  (une con mean_rgb/mean_audio por 'pid')")


if __name__ == "__main__":
    main()
