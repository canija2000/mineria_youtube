"""Grupos HDBSCAN sobre el mapa 3D y su nombre zero-shot.

Es el mismo procedimiento de `notebooks/05_clustering_hdbscan.ipynb`, aplicado a cada
versión del mapa para que la web pueda colorear por grupo:

1. HDBSCAN sobre las coordenadas 3D del mapa (los puntos sueltos quedan como ruido).
2. Centroide normalizado de cada grupo en el espacio original (768-d); el ruido se
   asigna al centroide más parecido.
3. Nombre de cada grupo: las 2 etiquetas de YT-8M y los 2 tags (folksonomía) más
   cercanos a su centroide, sin repetir.

Se guarda junto al mapa, en `data/maps*/<versión>/`:
  clusters.parquet   youtube_id, cluster (0..K-1), core (False si era ruido)
  clusters.json      [{id, name, size}] ordenado por id
  centroids.npy      (K, 768) para asignar grupo a videos que llegan después

Uso (para un mapa ya calculado; `remap.py` lo hace solo al publicar uno nuevo):
  python3 -m recolect.clusters               # mapa de texto vigente
  python3 -m recolect.clusters --rep image   # mapa de miniaturas vigente
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CLUSTER_SIZE = 220   # grupos grandes y legibles
MIN_SAMPLES = 30         # más alto = más puntos quedan como ruido
# El mapa de miniaturas es más continuo: con "eom" HDBSCAN lo deja casi en un solo grupo,
# así que ahí se eligen las hojas del árbol de densidad (~40 grupos, como en texto).
SELECTION = {"text": "eom", "image": "leaf"}
GENERIC_TAGS = {"video", "videos", "youtube", "new", "the", "and", "for", "with", "how", "live", "full",
                "free", "hd", "official", "part", "episode", "best", "top", "clip", "show"}


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def folksonomy(ids: list[str], vectors: np.ndarray, data_dir) -> tuple[np.ndarray, np.ndarray]:
    """Prototipo de cada tag frecuente: el centroide de los videos que lo usan (8–800 videos)."""
    position = {video_id: i for i, video_id in enumerate(ids)}
    tags = pd.read_parquet(Path(data_dir) / "yt8m_hydrated.parquet", columns=["youtube_id", "tags"])
    rows_of = defaultdict(list)
    for video_id, video_tags in zip(tags["youtube_id"], tags["tags"]):
        i = position.get(video_id)
        if i is None or not isinstance(video_tags, (list, np.ndarray)):
            continue
        for tag in {str(t).strip().lower() for t in video_tags}:
            if 3 <= len(tag) <= 28 and tag not in GENERIC_TAGS and not tag.isdigit():
                rows_of[tag].append(i)
    rows_of = {tag: rows for tag, rows in rows_of.items() if 8 <= len(rows) <= 800}
    names = np.array(list(rows_of))
    prototypes = _normalize(np.stack([vectors[rows].mean(axis=0) for rows in rows_of.values()]))
    return names, prototypes.astype(np.float32)


def cluster_map(ids: list[str], vectors: np.ndarray, coords: np.ndarray, data_dir="data", representation="text"):
    """Agrupa un mapa y devuelve (tabla por video, centroides, resumen por grupo)."""
    from sklearn.cluster import HDBSCAN

    raw = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                  cluster_selection_method=SELECTION[representation]).fit_predict(coords.astype(np.float64))
    k = int(raw.max()) + 1
    centroids = _normalize(np.stack([vectors[raw == c].mean(axis=0) for c in range(k)])).astype(np.float32)
    assigned = raw.copy()
    noise = raw == -1
    assigned[noise] = np.argmax(vectors[noise] @ centroids.T, axis=1)

    vocab = pd.read_parquet(Path(data_dir) / "emb_vocab.parquet", columns=["Name", "emb"])
    vocab_names = vocab["Name"].astype(str).to_numpy()
    vocab_vectors = np.stack(vocab["emb"].to_numpy()).astype(np.float32)
    folk_names, folk_vectors = folksonomy(ids, vectors, data_dir)

    summary = []
    for c in range(k):
        candidates = [*vocab_names[np.argsort(-(vocab_vectors @ centroids[c]))[:2]],
                      *folk_names[np.argsort(-(folk_vectors @ centroids[c]))[:2]]]
        words = []
        for word in candidates:
            if word.lower() not in {w.lower() for w in words}:
                words.append(word)
        summary.append({"id": c, "name": " · ".join(words[:3]), "size": int((assigned == c).sum())})

    table = pd.DataFrame({"youtube_id": ids, "cluster": assigned.astype(np.int16), "core": ~noise})
    return table, centroids, summary


def save(folder: Path, table: pd.DataFrame, centroids: np.ndarray, summary: list[dict]) -> None:
    table.to_parquet(folder / "clusters.parquet", index=False)
    np.save(folder / "centroids.npy", centroids)
    (folder / "clusters.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def load(folder: Path):
    """(tabla, centroides, resumen) de una versión del mapa, o None si no tiene grupos."""
    if not (folder / "clusters.parquet").exists():
        return None
    return (pd.read_parquet(folder / "clusters.parquet"), np.load(folder / "centroids.npy"),
            json.loads((folder / "clusters.json").read_text()))


def cluster_current(data_dir="data", representation="text") -> None:
    """Agrega los grupos al mapa vigente, sin recalcular el UMAP."""
    from .mapping import load_base_vectors, load_current, maps_dir
    from .received import ReceivedStore, store_path

    meta, coords, _reducer = load_current(data_dir, representation)
    base_ids, base_vectors = load_base_vectors(data_dir, representation)
    vector_of = dict(zip(base_ids, base_vectors))
    if representation == "text":
        vector_of.update({r["youtube_id"]: r["vector"] for r in ReceivedStore(store_path(data_dir)).all()})
    ids = coords["youtube_id"].astype(str).tolist()
    vectors = np.stack([vector_of[i] for i in ids]).astype(np.float32)
    table, centroids, summary = cluster_map(ids, vectors, coords[["x", "y", "z"]].to_numpy(), data_dir, representation)
    folder = maps_dir(data_dir, representation) / meta["version"]
    save(folder, table, centroids, summary)
    noise = 1 - table["core"].mean()
    print(f"[grupos {representation}] {len(summary)} grupos en {meta['version']} "
          f"({noise:.0%} de ruido asignado al centroide más cercano) -> {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="carpeta de datos (data o data/dev)")
    parser.add_argument("--rep", default="text", choices=("text", "image"), help="mapa de texto o de miniaturas")
    args = parser.parse_args()
    cluster_current(args.data, args.rep)


if __name__ == "__main__":
    main()
