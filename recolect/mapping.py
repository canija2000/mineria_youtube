"""Mapa 3D (UMAP) guardado y versionado.

- El mapa se calcula aquí, no en el servidor. El servidor solo lo carga.
- Cada cálculo crea una versión nueva en `data/maps/<versión>/` y `data/maps/current.json`
  indica cuál está vigente. El cambio de versión es atómico: el servidor nunca ve un mapa a medias.
- El mapa de texto incluye los videos base (YT-8M) y los recibidos por el servidor hasta ese momento.
- El mapa de imagen (`data/maps_image/`) usa las miniaturas de los videos base. SQLite no guarda
  vectores de imagen, así que los recibidos se ubican sobre él al consultarlos.

Uso (rutina de recálculo):
  python3 remap.py                 # recalcula con todos los videos base + recibidos
  python3 remap.py --min-new 50    # solo recalcula si llegaron al menos 50 videos nuevos
  python3 remap.py --rep image     # recalcula el mapa de miniaturas
  python3 remap.py --data data/dev # sobre los datos de prueba
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .received import ReceivedStore, store_path

REPRESENTATIONS = ("text", "image")
UMAP_PARAMS = {"n_components": 3, "metric": "cosine", "n_neighbors": 15, "min_dist": 0.1, "random_state": 42}


def maps_dir(data_dir="data", representation="text") -> Path:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Representación desconocida: {representation}")
    return Path(data_dir) / ("maps" if representation == "text" else f"maps_{representation}")


def load_base_vectors(data_dir="data", representation="text"):
    """Vectores de los videos base (texto o miniatura), sin IDs vacíos ni repetidos."""
    frame = pd.read_parquet(Path(data_dir) / f"emb_{representation}.parquet", columns=["youtube_id", "emb"])
    frame = frame.dropna(subset=["youtube_id"]).drop_duplicates("youtube_id").reset_index(drop=True)
    ids = frame["youtube_id"].astype(str).tolist()
    vectors = np.stack(frame["emb"].to_numpy()).astype(np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 768:
        raise ValueError(f"Se esperaban embeddings (n, 768); recibido {vectors.shape}")
    return ids, vectors


def current_info(data_dir="data", representation="text") -> dict | None:
    """Metadatos del mapa vigente, o None si todavía no hay mapa."""
    pointer = maps_dir(data_dir, representation) / "current.json"
    if not pointer.exists():
        return None
    version = json.loads(pointer.read_text())["version"]
    meta_path = maps_dir(data_dir, representation) / version / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def load_current(data_dir="data", representation="text"):
    """Devuelve (meta, coordenadas por youtube_id, reducer) del mapa vigente."""
    import joblib

    meta = current_info(data_dir, representation)
    if meta is None:
        raise FileNotFoundError(f"No hay mapa guardado en {maps_dir(data_dir, representation)}. "
                                f"Calcúlalo con: python3 remap.py --rep {representation} --data {data_dir}")
    folder = maps_dir(data_dir, representation) / meta["version"]
    coords = pd.read_parquet(folder / "coords.parquet")
    reducer = joblib.load(folder / "reducer.joblib")
    return meta, coords, reducer


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def _prune(folder: Path, keep: int, current_version: str) -> None:
    """Deja solo las `keep` versiones más recientes (nunca borra la vigente)."""
    if keep <= 0:
        return
    versions = sorted(p for p in folder.iterdir()
                      if p.is_dir() and not p.name.startswith("."))
    for folder in versions[:-keep]:
        if folder.name != current_version:
            shutil.rmtree(folder)
            print(f"[mapa] versión antigua eliminada: {folder.name}")


def build_map(data_dir="data", min_new: int = 0, keep: int = 3, representation="text") -> dict | None:
    """Recalcula el mapa con videos base + recibidos y lo deja como versión vigente.

    Si `min_new` > 0 y llegaron menos videos nuevos desde el último mapa, no hace nada.
    """
    import joblib
    import umap

    base_ids, base_vectors = load_base_vectors(data_dir, representation)
    base_set = set(base_ids)
    received = []
    if representation == "text":
        received = [r for r in ReceivedStore(store_path(data_dir)).all() if r["youtube_id"] not in base_set]

    previous = current_info(data_dir, representation)
    new_since_last = len(received) - (previous["n_received"] if previous else 0)
    if previous and new_since_last < min_new:
        print(f"[mapa] {new_since_last} videos nuevos desde la versión {previous['version']} "
              f"(mínimo {min_new}); no se recalcula")
        return None

    ids = base_ids + [r["youtube_id"] for r in received]
    vectors = base_vectors
    if received:
        vectors = np.vstack([base_vectors, np.stack([r["vector"] for r in received])]).astype(np.float32)
    sources = ["yt8m"] * len(base_ids) + ["recibido"] * len(received)

    print(f"[mapa {representation}] UMAP 3D sobre {len(base_ids):,} videos base + {len(received):,} recibidos...", flush=True)
    started = datetime.now(timezone.utc)
    reducer = umap.UMAP(**UMAP_PARAMS)
    coordinates = reducer.fit_transform(vectors).astype(np.float32)

    version = started.strftime("%Y%m%d-%H%M%S")
    root = maps_dir(data_dir, representation)
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".tmp-{version}"
    staging.mkdir()
    pd.DataFrame({"youtube_id": ids, "x": coordinates[:, 0], "y": coordinates[:, 1],
                  "z": coordinates[:, 2], "source": sources}).to_parquet(staging / "coords.parquet", index=False)
    joblib.dump(reducer, staging / "reducer.joblib")
    meta = {
        "version": version,
        "created_at": started.isoformat(),
        "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "representation": representation,
        "umap": UMAP_PARAMS,
        "n_base": len(base_ids),
        "n_received": len(received),
    }
    (staging / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    staging.rename(root / version)
    _write_json_atomic(root / "current.json", {"version": version})
    print(f"[mapa] versión {version} vigente ({len(ids):,} puntos, {meta['seconds']}s) -> {root / version}")
    _prune(root, keep, version)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="carpeta de datos (data o data/dev)")
    parser.add_argument("--min-new", type=int, default=0,
                        help="recalcula solo si llegaron al menos N videos nuevos desde el último mapa")
    parser.add_argument("--keep", type=int, default=3, help="versiones de mapa a conservar (0 = todas)")
    parser.add_argument("--rep", default="text", choices=REPRESENTATIONS, help="mapa de texto o de miniaturas")
    args = parser.parse_args()
    build_map(args.data, min_new=args.min_new, keep=args.keep, representation=args.rep)


if __name__ == "__main__":
    main()
