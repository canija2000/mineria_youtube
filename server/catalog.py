"""Catálogo del servidor: mapa guardado + videos base + videos recibidos.

- El mapa (posiciones 3D y el "molde" UMAP para ubicar videos nuevos) se calcula aparte
  con `python3 remap.py` o `python3 recolect.py`; aquí solo se carga.
- Cada video recibido se guarda en SQLite con su vector, sin posición fija: su posición
  se calcula sobre el mapa vigente. Así, cuando `remap.py` publica un mapa nuevo,
  el servidor lo detecta y reubica todo sin reiniciarse.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

from recolect import clusters as map_clusters
from recolect.mapping import current_info, load_base_vectors, load_current, maps_dir
from recolect.received import ReceivedStore, store_path


def _clean(value):
    return "" if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)) else value


def _clean_num(value):
    if value is None or (not isinstance(value, (int, float)) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (ValueError, TypeError):
        return None


def _card(metadata: dict) -> dict:
    desc = _clean(metadata.get("description"))
    if desc and len(desc) > 300:
        desc = desc[:300] + "..."
    return {
        "title": _clean(metadata.get("title")),
        "channel_title": _clean(metadata.get("channel_title")),
        "thumbnail_url": _clean(metadata.get("thumbnail_url")),
        "view_count": _clean_num(metadata.get("view_count")),
        "like_count": _clean_num(metadata.get("like_count")),
        "duration": _clean(metadata.get("duration")),
        "published_at": _clean(metadata.get("published_at")),
        "description": desc,
    }


class VideoCatalog:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.lock = RLock()
        self.store = ReceivedStore(store_path(self.data_dir))

        self.base_ids, self.base_vectors = load_base_vectors(self.data_dir)
        self.base_set = set(self.base_ids)
        cols = [
            "youtube_id", "title", "channel_title", "thumbnail_url",
            "view_count", "like_count", "duration", "published_at",
            "description", "labels", "tags"
        ]
        metadata = pd.read_parquet(self.data_dir / "yt8m_hydrated.parquet", columns=cols)
        metadata = metadata.dropna(subset=["youtube_id"])
        self.base_metadata = {str(row.youtube_id): _card(row._asdict()) for row in metadata.itertuples()}
        self.vocab_names = None
        self.vocab_vectors = None
        self.folk_names = None
        self.folk_vectors = None
        self.cat_names: list[str] = []
        self.cat_centroids: np.ndarray | None = None
        self._last_store_count: int = 0
        self.verticals = self._load_verticals(metadata)
        self._build_category_centroids()
        self._build_folksonomy_vocab(metadata)

        self.image_ids: list[str] = []
        self.image_positions: dict[str, int] = {}
        self.image_vectors: np.ndarray | None = None
        self.image_coordinates: np.ndarray | None = None
        self.image_meta: dict = {}
        self.image_cluster_ids: np.ndarray | None = None
        self.image_cluster_names: list[str] = []
        self._load_image_space()

        self.map_meta: dict = {}
        self._map_cache = None
        self._image_map_cache = None
        self._load_map()

    def _load_verticals(self, metadata: pd.DataFrame) -> dict[str, str]:
        """Categoría general (vertical) de cada video base: la más frecuente entre sus etiquetas de YT-8M.

        Es el mismo criterio de `analysis/metricas.py::categoria` (mode(Vertical1)). Los videos recibidos no tienen
        etiquetas de YT-8M, así que no tienen vertical.
        """
        vocab_path = self.data_dir / "emb_vocab.parquet"
        if not vocab_path.exists():
            vocab_path = Path("data") / "emb_vocab.parquet"
        if vocab_path.exists():
            vocab = pd.read_parquet(vocab_path, columns=["Index", "Name", "Vertical1", "emb"])
            self.vocab_names = vocab["Name"].astype(str).to_numpy()
            self.vocab_vectors = np.stack(vocab["emb"].to_numpy()).astype(np.float32)
        else:
            vocab = pd.read_csv(Path("data") / "yt8m_vocabulary.csv", usecols=["Index", "Vertical1"])
        label_vertical = dict(zip(vocab["Index"].astype(int), vocab["Vertical1"]))
        verticals = {}
        for video_id, labels in zip(metadata["youtube_id"].astype(str), metadata["labels"]):
            names = [label_vertical[int(label)] for label in (labels if labels is not None else [])
                     if int(label) in label_vertical and isinstance(label_vertical[int(label)], str)]
            if names:
                verticals[video_id] = Counter(names).most_common(1)[0][0]
        return verticals

    def _build_category_centroids(self) -> None:
        """Calcula el centroide normalizado de cada categoría sobre los videos base para clasificar videos recibidos."""
        cat_rows = defaultdict(list)
        for i, yid in enumerate(self.base_ids):
            v = self.verticals.get(yid)
            if v and v != "(Unknown)":
                cat_rows[v].append(i)
        if not cat_rows:
            return
        self.cat_names = sorted(cat_rows.keys())
        mat = np.zeros((len(self.cat_names), self.base_vectors.shape[1]), dtype=np.float32)
        for k, name in enumerate(self.cat_names):
            c = self.base_vectors[cat_rows[name]].mean(axis=0)
            c /= max(float(np.linalg.norm(c)), 1e-12)
            mat[k] = c
        self.cat_centroids = mat

    def _classify_vector_vertical(self, vec: np.ndarray) -> str | None:
        if self.cat_centroids is None or not self.cat_names:
            return None
        sims = self.cat_centroids @ vec
        return self.cat_names[int(np.argmax(sims))]

    def _build_folksonomy_vocab(self, metadata: pd.DataFrame, max_concepts: int = 6000) -> None:
        """Vocabulario extendido (folksonomía) a partir de los prototipos semánticos de tags recurrentes."""
        if "tags" not in metadata.columns:
            return
        stop = {
            "video", "videos", "youtube", "new", "the", "and", "for", "with", "how",
            "live", "full", "free", "hd", "official", "2011", "2012", "2013", "2014",
            "2015", "2016", "part", "episode", "best", "top", "clip", "show"
        }
        id_to_idx = {yid: i for i, yid in enumerate(self.base_ids)}
        tag_rows = defaultdict(list)
        for yid, tags in zip(metadata["youtube_id"].astype(str), metadata["tags"]):
            idx = id_to_idx.get(yid)
            if idx is None or tags is None:
                continue
            for raw in set(str(x).strip().lower() for x in tags):
                if 3 <= len(raw) <= 28 and raw not in stop and not raw.isdigit():
                    tag_rows[raw].append(idx)
        frequent = [(t, rows) for t, rows in tag_rows.items() if 8 <= len(rows) <= 800]
        if not frequent:
            return
        frequent.sort(key=lambda x: -len(x[1]))
        frequent = frequent[:max_concepts]
        self.folk_names = np.array([t for t, _ in frequent])
        mat = np.zeros((len(frequent), self.base_vectors.shape[1]), dtype=np.float32)
        for k, (_, rows) in enumerate(frequent):
            v = self.base_vectors[rows].mean(axis=0)
            v /= max(float(np.linalg.norm(v)), 1e-12)
            mat[k] = v
        self.folk_vectors = mat

    def _predict_zero_shot(self, vec: np.ndarray) -> list[dict]:
        results = []
        seen = set()
        if self.vocab_vectors is not None and self.vocab_names is not None:
            sims = self.vocab_vectors @ vec
            for j in np.argsort(-sims)[:5]:
                name = str(self.vocab_names[j])
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    results.append({"tag": name, "score": round(float(sims[j]), 3), "source": "yt8m"})
        if self.folk_vectors is not None and self.folk_names is not None:
            sims = self.folk_vectors @ vec
            added = 0
            for j in np.argsort(-sims)[:12]:
                name = str(self.folk_names[j])
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    results.append({"tag": name, "score": round(float(sims[j]), 3), "source": "ext"})
                    added += 1
                    if added >= 6:
                        break
        return results

    # ---------- carga del mapa y armado del catálogo ----------

    def _load_image_space(self) -> None:
        """Carga los vectores de miniatura de los videos base y su mapa UMAP 3D (`remap.py --rep image`)."""
        if current_info(self.data_dir, "image") is None:
            print("[catálogo] sin mapa de miniaturas; calcúlalo con: python3 remap.py --rep image", flush=True)
            return
        meta, coords, _reducer = load_current(self.data_dir, "image")
        ids, vectors = load_base_vectors(self.data_dir, "image")
        coords = coords.set_index("youtube_id").reindex(ids)
        keep = coords["x"].notna().to_numpy()
        self.image_meta = meta
        self.image_ids = [video_id for video_id, ok in zip(ids, keep) if ok]
        self.image_positions = {yid: i for i, yid in enumerate(self.image_ids)}
        self.image_vectors = vectors[keep]
        self.image_coordinates = coords.loc[keep, ["x", "y", "z"]].to_numpy(dtype=np.float32)
        loaded = map_clusters.load(maps_dir(self.data_dir, "image") / meta["version"])
        if loaded:
            table, _centroids, summary = loaded
            cluster_of = dict(zip(table["youtube_id"], table["cluster"]))
            self.image_cluster_ids = np.array([cluster_of.get(i, -1) for i in self.image_ids], dtype=np.int16)
            self.image_cluster_names = [c["name"] for c in sorted(summary, key=lambda c: c["id"])]
        print(f"[catálogo] mapa de miniaturas {meta['version']}: {len(self.image_ids):,} videos", flush=True)

    def _load_map(self):
        meta, coords, reducer = load_current(self.data_dir)
        self.map_meta, self.reducer = meta, reducer
        # Grupos HDBSCAN de esta versión (recolect/clusters.py); None si el mapa no los tiene
        self.map_clusters = map_clusters.load(maps_dir(self.data_dir) / meta["version"])
        self.map_coords = {video_id: np.array([x, y, z], dtype=np.float32)
                           for video_id, x, y, z in coords[["youtube_id", "x", "y", "z"]].itertuples(index=False)}
        self._rebuild()
        print(f"[catálogo] mapa {meta['version']}: {len(self.ids):,} videos "
              f"({self.pending_count} recibidos aún no incluidos en el mapa)", flush=True)

    def _rebuild(self):
        all_received = self.store.all()
        self._last_store_count = len(all_received)
        received = [r for r in all_received if r["youtube_id"] not in self.base_set]
        self.ids = self.base_ids + [r["youtube_id"] for r in received]
        self.sources = ["yt8m"] * len(self.base_ids) + ["recibido"] * len(received)
        self.vectors = self.base_vectors
        if received:
            rec_vecs = np.stack([r["vector"] for r in received]).astype(np.float32)
            self.vectors = np.vstack([self.base_vectors, rec_vecs])
            # Asignar categoría estimada por centroide a los videos recibidos para que cuenten en las categorías del mapa
            if self.cat_centroids is not None and self.cat_names:
                cat_idx = np.argmax(rec_vecs @ self.cat_centroids.T, axis=1)
                for r, c_i in zip(received, cat_idx):
                    yid = r["youtube_id"]
                    if yid not in self.verticals:
                        self.verticals[yid] = self.cat_names[int(c_i)]
        self.metadata = dict(self.base_metadata)
        self.metadata.update({r["youtube_id"]: _card(r["metadata"]) for r in received})
        self.positions = {video_id: i for i, video_id in enumerate(self.ids)}

        # Posiciones: las del mapa guardado; lo que no está en el mapa se ubica con el molde UMAP.
        self.in_map = np.array([video_id in self.map_coords for video_id in self.ids])
        self.coordinates = np.zeros((len(self.ids), 3), dtype=np.float32)
        for i, video_id in enumerate(self.ids):
            if self.in_map[i]:
                self.coordinates[i] = self.map_coords[video_id]
        missing = np.flatnonzero(~self.in_map)
        if len(missing):
            self.coordinates[missing] = self.reducer.transform(self.vectors[missing]).astype(np.float32)

        # Grupo HDBSCAN: el del mapa; lo que llegó después va al centroide más parecido.
        self.cluster_ids = np.full(len(self.ids), -1, dtype=np.int16)
        self.cluster_names: list[str] = []
        if self.map_clusters:
            table, centroids, summary = self.map_clusters
            self.cluster_names = [c["name"] for c in sorted(summary, key=lambda c: c["id"])]
            cluster_of = dict(zip(table["youtube_id"], table["cluster"]))
            for i, video_id in enumerate(self.ids):
                self.cluster_ids[i] = cluster_of.get(video_id, -1)
            unassigned = np.flatnonzero(self.cluster_ids < 0)
            if len(unassigned):
                self.cluster_ids[unassigned] = np.argmax(self.vectors[unassigned] @ centroids.T, axis=1)

    def _cluster_for(self, vector: np.ndarray) -> int:
        if not self.map_clusters:
            return -1
        return int(np.argmax(self.map_clusters[1] @ vector))

    def refresh_map(self):
        """Si `remap.py` publicó un mapa nuevo o entraron lotes en SQLite (`--modern`), actualiza el catálogo."""
        latest_image = current_info(self.data_dir, "image")
        if latest_image and latest_image["version"] != self.image_meta.get("version"):
            with self.lock:
                self._load_image_space()
                self._image_map_cache = None
        latest = current_info(self.data_dir)
        store_count = self.store.count()
        if (latest and latest["version"] != self.map_meta["version"]) or (store_count != self._last_store_count):
            with self.lock:
                if latest and latest["version"] != self.map_meta["version"]:
                    self._load_map()
                elif self.store.count() != self._last_store_count:
                    self._rebuild()

    # ---------- consultas ----------

    @property
    def pending_count(self) -> int:
        """Videos recibidos que todavía no forman parte del mapa (se ubicaron sobre él)."""
        return int(sum(1 for i, source in enumerate(self.sources) if source == "recibido" and not self.in_map[i]))

    def status(self) -> dict:
        return {"map_version": self.map_meta["version"], "map_created_at": self.map_meta["created_at"],
                "videos": len(self.ids), "received": self.sources.count("recibido"),
                "received_not_in_map": self.pending_count}

    def has(self, video_id):
        return video_id in self.positions

    def map_payload(self, representation: str = "text") -> dict:
        """Todos los puntos del mapa vigente, en columnas para que la respuesta pese poco.

        `vertical` y `source` son índices dentro de las listas `verticals` y `sources`
        (vertical = -1 si el video no tiene categoría, por ejemplo los recibidos).
        Se guarda en memoria hasta que cambie el mapa o llegue un video nuevo.
        """
        with self.lock:
            if representation == "image" and self.image_ids and self.image_coordinates is not None:
                if self._image_map_cache and self._image_map_cache[0] == self.image_meta["version"]:
                    return self._image_map_cache[1]
                vertical_counts = Counter(
                    self.verticals.get(i) for i in self.image_ids
                    if self.verticals.get(i) and self.verticals.get(i) != "(Unknown)"
                )
                vertical_names = [name for name, _ in vertical_counts.most_common() if name]
                if any(self.verticals.get(i) == "(Unknown)" for i in self.image_ids):
                    vertical_names.append("(Unknown)")
                vertical_index = {name: i for i, name in enumerate(vertical_names)}
                coords = np.round(self.image_coordinates.astype(float), 4)
                payload = {
                    "map_version": self.image_meta["version"],
                    "representation": "image",
                    "videos": len(self.image_ids),
                    "verticals": vertical_names,
                    "sources": ["yt8m"],
                    "points": {
                        "youtube_id": list(self.image_ids),
                        "x": coords[:, 0].tolist(),
                        "y": coords[:, 1].tolist(),
                        "z": coords[:, 2].tolist(),
                        "title": [self.metadata.get(i, {}).get("title", "") for i in self.image_ids],
                        "channel_title": [self.metadata.get(i, {}).get("channel_title", "") for i in self.image_ids],
                        "vertical": [vertical_index.get(self.verticals.get(i, ""), -1) for i in self.image_ids],
                        "source": [0] * len(self.image_ids),
                        "in_map": [True] * len(self.image_ids),
                        "cluster": (self.image_cluster_ids.tolist() if self.image_cluster_ids is not None
                                    else [-1] * len(self.image_ids)),
                    },
                    "clusters": self.image_cluster_names,
                }
                self._image_map_cache = (self.image_meta["version"], payload)
                return payload

            key = (self.map_meta["version"], len(self.ids))
            if self._map_cache and self._map_cache[0] == key:
                return self._map_cache[1]
            vertical_counts = Counter(v for v in self.verticals.values() if v and v != "(Unknown)")
            vertical_names = [name for name, _ in vertical_counts.most_common()]
            if "(Unknown)" in set(self.verticals.values()):
                vertical_names.append("(Unknown)")
            vertical_index = {name: i for i, name in enumerate(vertical_names)}
            source_names = ["yt8m", "recibido"]
            coordinates = np.round(self.coordinates.astype(float), 4)
            payload = {
                **self.status(),
                "verticals": vertical_names,
                "sources": source_names,
                "points": {
                    "youtube_id": list(self.ids),
                    "x": coordinates[:, 0].tolist(),
                    "y": coordinates[:, 1].tolist(),
                    "z": coordinates[:, 2].tolist(),
                    "title": [self.metadata.get(i, {}).get("title", "") for i in self.ids],
                    "channel_title": [self.metadata.get(i, {}).get("channel_title", "") for i in self.ids],
                    "vertical": [vertical_index.get(self.verticals.get(i), -1) for i in self.ids],
                    "source": [source_names.index(s) for s in self.sources],
                    "in_map": self.in_map.astype(bool).tolist(),
                    "cluster": self.cluster_ids.tolist(),
                },
                "clusters": self.cluster_names,
            }
            self._map_cache = (key, payload)
            return payload

    def add(self, video_id, metadata, vector):
        """Guarda el video recibido, lo ubica sobre el mapa vigente y devuelve su punto y vecinos."""
        with self.lock:
            if self.has(video_id):
                return self.describe(video_id)
            vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            coordinate = self.reducer.transform(vector.reshape(1, -1))[0].astype(np.float32)
            self.store.add(video_id, metadata, vector)
            self.positions[video_id] = len(self.ids)
            self.ids.append(video_id)
            self.sources.append("recibido")
            self.metadata[video_id] = _card(metadata)
            pred_vert = self._classify_vector_vertical(vector)
            if pred_vert and video_id not in self.verticals:
                self.verticals[video_id] = pred_vert
            self.vectors = np.vstack((self.vectors, vector.reshape(1, -1)))
            self.coordinates = np.vstack((self.coordinates, coordinate.reshape(1, -1)))
            self.in_map = np.append(self.in_map, False)
            self.cluster_ids = np.append(self.cluster_ids, np.int16(self._cluster_for(vector)))
            # Solo cuenta el video recién agregado: si entró un lote por fuera (`--modern`),
            # el total de SQLite sigue distinto y el próximo refresh_map() lo carga.
            self._last_store_count += 1
            return self.describe(video_id)

    def describe(self, video_id, neighbor_count=12):
        with self.lock:
            position = self.positions[video_id]
            similarities = self.vectors @ self.vectors[position]

            nearest_sims = similarities.copy()
            nearest_sims[position] = -np.inf
            nearest = np.argsort(-nearest_sims)[:neighbor_count]

            farthest_sims = similarities.copy()
            farthest_sims[position] = np.inf
            far_pool = np.argsort(farthest_sims)[:400]
            farthest = []
            seen_verticals = set()
            for i in far_pool:
                v = self.verticals.get(self.ids[i])
                if v and v not in seen_verticals:
                    seen_verticals.add(v)
                    farthest.append(i)
                    if len(farthest) >= neighbor_count:
                        break
            if len(farthest) < neighbor_count:
                chosen = set(farthest)
                for i in far_pool:
                    if i not in chosen:
                        farthest.append(i)
                        if len(farthest) >= neighbor_count:
                            break

            neighbor_verticals = [
                self.verticals.get(self.ids[i])
                for i in nearest
                if self.verticals.get(self.ids[i])
            ]
            vote_counts = Counter(neighbor_verticals).most_common()
            predicted_vertical = vote_counts[0][0] if vote_counts else None
            knn_votes = {
                "vertical": predicted_vertical,
                "count": vote_counts[0][1] if vote_counts else 0,
                "total": len(neighbor_verticals),
                "breakdown": [{"vertical": v, "count": c} for v, c in vote_counts],
            }

            zero_shot_tags = self._predict_zero_shot(self.vectors[position])

            return {
                "map_version": self.map_meta["version"],
                "video": {
                    "youtube_id": video_id,
                    **self.metadata.get(video_id, {}),
                    "vertical": self.verticals.get(video_id),
                    "predicted_vertical": predicted_vertical,
                    "knn_votes": knn_votes,
                    "zero_shot_tags": zero_shot_tags,
                    "source": self.sources[position],
                    "in_map": bool(self.in_map[position]),
                },
                "projection": self.coordinates[position].astype(float).tolist(),
                "neighbors": [
                    {"youtube_id": self.ids[i], **self.metadata.get(self.ids[i], {}),
                     "vertical": self.verticals.get(self.ids[i]),
                     "similarity": float(similarities[i]),
                     "projection": self.coordinates[i].astype(float).tolist()}
                    for i in nearest
                ],
                "farthest": [
                    {"youtube_id": self.ids[i], **self.metadata.get(self.ids[i], {}),
                     "vertical": self.verticals.get(self.ids[i]),
                     "similarity": float(similarities[i]),
                     "projection": self.coordinates[i].astype(float).tolist()}
                    for i in farthest
                ],
            }

    def describe_image(
        self,
        video_id: str,
        metadata: dict | None = None,
        image_vector: np.ndarray | None = None,
        neighbor_count: int = 12,
    ) -> dict:
        """Describe un video en el espacio visual (miniatura) y devuelve sus vecinos y lejanos visuales."""
        with self.lock:
            if self.image_vectors is None or self.image_coordinates is None or not self.image_ids:
                return self.describe(video_id, neighbor_count=neighbor_count)

            if metadata is not None and video_id not in self.metadata:
                self.metadata[video_id] = _card(metadata)

            in_image_map = video_id in self.image_positions
            if in_image_map:
                pos = self.image_positions[video_id]
                vec = self.image_vectors[pos]
                coord = self.image_coordinates[pos].astype(float).tolist()
                similarities = self.image_vectors @ vec
                nearest_sims = similarities.copy()
                nearest_sims[pos] = -np.inf
                nearest = np.argsort(-nearest_sims)[:neighbor_count]
                farthest_sims = similarities.copy()
                farthest_sims[pos] = np.inf
            elif image_vector is not None:
                vec = np.asarray(image_vector, dtype=np.float32).reshape(-1)
                vec /= max(float(np.linalg.norm(vec)), 1e-12)
                similarities = self.image_vectors @ vec
                nearest = np.argsort(-similarities)[:neighbor_count]
                top_k = nearest[:15]
                weights = np.clip(similarities[top_k], 0.01, None)
                weights /= weights.sum()
                coord = (self.image_coordinates[top_k] * weights[:, None]).sum(axis=0).astype(float).tolist()
                farthest_sims = similarities.copy()
            else:
                return self.describe(video_id, neighbor_count=neighbor_count)

            far_pool = np.argsort(farthest_sims)[:400]
            farthest = []
            seen_verticals = set()
            for i in far_pool:
                v = self.verticals.get(self.image_ids[i])
                if v and v not in seen_verticals:
                    seen_verticals.add(v)
                    farthest.append(i)
                    if len(farthest) >= neighbor_count:
                        break
            if len(farthest) < neighbor_count:
                chosen = set(farthest)
                for i in far_pool:
                    if i not in chosen:
                        farthest.append(i)
                        if len(farthest) >= neighbor_count:
                            break

            neighbor_verticals = [
                self.verticals.get(self.image_ids[i])
                for i in nearest
                if self.verticals.get(self.image_ids[i])
            ]
            vote_counts = Counter(neighbor_verticals).most_common()
            predicted_vertical = vote_counts[0][0] if vote_counts else None
            knn_votes = {
                "vertical": predicted_vertical,
                "count": vote_counts[0][1] if vote_counts else 0,
                "total": len(neighbor_verticals),
                "breakdown": [{"vertical": v, "count": c} for v, c in vote_counts],
            }

            return {
                "map_version": self.image_meta.get("version"),
                "representation": "image",
                "video": {
                    "youtube_id": video_id,
                    **self.metadata.get(video_id, {}),
                    "vertical": self.verticals.get(video_id),
                    "predicted_vertical": predicted_vertical,
                    "knn_votes": knn_votes,
                    "zero_shot_tags": self._predict_zero_shot(vec),
                    "source": "yt8m" if video_id in self.base_set else "recibido",
                    "in_map": in_image_map,
                },
                "projection": coord,
                "neighbors": [
                    {
                        "youtube_id": self.image_ids[i],
                        **self.metadata.get(self.image_ids[i], {}),
                        "vertical": self.verticals.get(self.image_ids[i]),
                        "similarity": float(similarities[i]),
                        "projection": self.image_coordinates[i].astype(float).tolist(),
                    }
                    for i in nearest
                ],
                "farthest": [
                    {
                        "youtube_id": self.image_ids[i],
                        **self.metadata.get(self.image_ids[i], {}),
                        "vertical": self.verticals.get(self.image_ids[i]),
                        "similarity": float(similarities[i]),
                        "projection": self.image_coordinates[i].astype(float).tolist(),
                    }
                    for i in farthest
                ],
            }

    def locate_concept(self, concept: str, vector: np.ndarray, neighbor_count: int = 12) -> dict:
        """Ubica un concepto o palabra libre en el mapa 3D y devuelve su región semántica y videos asociados."""
        with self.lock:
            q = np.asarray(vector, dtype=np.float32).reshape(-1)
            q /= max(float(np.linalg.norm(q)), 1e-12)
            key = concept.strip().lower()

            # Si el concepto coincide exactamente con el vocabulario o la folksonomía, combinamos con su prototipo
            if self.folk_names is not None and self.folk_vectors is not None:
                matches = np.flatnonzero(self.folk_names == key)
                if len(matches):
                    q = 0.5 * q + 0.5 * self.folk_vectors[matches[0]]
                    q /= max(float(np.linalg.norm(q)), 1e-12)
            if self.vocab_names is not None and self.vocab_vectors is not None:
                matches = [i for i, n in enumerate(self.vocab_names) if n.lower() == key]
                if matches:
                    q = 0.5 * q + 0.5 * self.vocab_vectors[matches[0]]
                    q /= max(float(np.linalg.norm(q)), 1e-12)

            similarities = self.vectors @ q
            nearest = np.argsort(-similarities)[:neighbor_count]
            region_idx = np.argsort(-similarities)[:30]

            # Centroide 3D ponderado por similitud sobre los top-20 videos del concepto
            top_k = region_idx[:20]
            weights = np.clip(similarities[top_k], 0.01, None)
            weights /= weights.sum()
            centroid = (self.coordinates[top_k] * weights[:, None]).sum(axis=0).astype(float).tolist()

            far_pool = np.argsort(similarities)[:400]
            farthest = []
            seen_verticals = set()
            for i in far_pool:
                v = self.verticals.get(self.ids[i])
                if v and v not in seen_verticals:
                    seen_verticals.add(v)
                    farthest.append(i)
                    if len(farthest) >= neighbor_count:
                        break

            neighbor_verticals = [
                self.verticals.get(self.ids[i])
                for i in region_idx
                if self.verticals.get(self.ids[i])
            ]
            vote_counts = Counter(neighbor_verticals).most_common()
            predicted_vertical = vote_counts[0][0] if vote_counts else None
            knn_votes = {
                "vertical": predicted_vertical,
                "count": vote_counts[0][1] if vote_counts else 0,
                "total": len(neighbor_verticals),
                "breakdown": [{"vertical": v, "count": c} for v, c in vote_counts],
            }

            return {
                "map_version": self.map_meta["version"],
                "concept": {
                    "query": concept.strip(),
                    "predicted_vertical": predicted_vertical,
                    "knn_votes": knn_votes,
                    "zero_shot_tags": self._predict_zero_shot(q),
                },
                "projection": centroid,
                "region_points": [self.coordinates[i].astype(float).tolist() for i in region_idx],
                "neighbors": [
                    {"youtube_id": self.ids[i], **self.metadata.get(self.ids[i], {}),
                     "vertical": self.verticals.get(self.ids[i]),
                     "similarity": float(similarities[i]),
                     "projection": self.coordinates[i].astype(float).tolist()}
                    for i in nearest
                ],
                "farthest": [
                    {"youtube_id": self.ids[i], **self.metadata.get(self.ids[i], {}),
                     "vertical": self.verticals.get(self.ids[i]),
                     "similarity": float(similarities[i]),
                     "projection": self.coordinates[i].astype(float).tolist()}
                    for i in farthest
                ],
            }

    def locate_external_image(
        self,
        image_vector: np.ndarray,
        label: str = "Imagen consultada",
        preview_url: str = "",
        neighbor_count: int = 12,
    ) -> dict:
        """Ubica una imagen arbitraria (archivo o URL) en el espacio visual SIN guardarla en la BDD."""
        with self.lock:
            vec = np.asarray(image_vector, dtype=np.float32).reshape(-1)
            vec /= max(float(np.linalg.norm(vec)), 1e-12)

            if self.image_vectors is not None and self.image_coordinates is not None and self.image_ids:
                ids_pool = self.image_ids
                coords_pool = self.image_coordinates
                similarities = self.image_vectors @ vec
            else:
                ids_pool = self.ids
                coords_pool = self.coordinates
                similarities = self.vectors @ vec

            nearest = np.argsort(-similarities)[:neighbor_count]
            top_k = nearest[:15]
            weights = np.clip(similarities[top_k], 0.01, None)
            weights /= weights.sum()
            coord = (coords_pool[top_k] * weights[:, None]).sum(axis=0).astype(float).tolist()

            far_pool = np.argsort(similarities)[:400]
            farthest = []
            seen_verticals = set()
            for i in far_pool:
                v = self.verticals.get(ids_pool[i])
                if v and v not in seen_verticals:
                    seen_verticals.add(v)
                    farthest.append(i)
                    if len(farthest) >= neighbor_count:
                        break
            if len(farthest) < neighbor_count:
                chosen = set(farthest)
                for i in far_pool:
                    if i not in chosen:
                        farthest.append(i)
                        if len(farthest) >= neighbor_count:
                            break

            neighbor_verticals = [
                self.verticals.get(ids_pool[i])
                for i in nearest
                if self.verticals.get(ids_pool[i])
            ]
            vote_counts = Counter(neighbor_verticals).most_common()
            predicted_vertical = vote_counts[0][0] if vote_counts else None
            knn_votes = {
                "vertical": predicted_vertical,
                "count": vote_counts[0][1] if vote_counts else 0,
                "total": len(neighbor_verticals),
                "breakdown": [{"vertical": v, "count": c} for v, c in vote_counts],
            }

            return {
                "map_version": self.image_meta.get("version"),
                "representation": "image",
                "external_image": {
                    "label": label,
                    "preview_url": preview_url,
                    "predicted_vertical": predicted_vertical,
                    "knn_votes": knn_votes,
                    "zero_shot_tags": self._predict_zero_shot(vec),
                    "persisted": False,
                },
                "projection": coord,
                "neighbors": [
                    {
                        "youtube_id": ids_pool[i],
                        **self.metadata.get(ids_pool[i], {}),
                        "vertical": self.verticals.get(ids_pool[i]),
                        "similarity": float(similarities[i]),
                        "projection": coords_pool[i].astype(float).tolist(),
                    }
                    for i in nearest
                ],
                "farthest": [
                    {
                        "youtube_id": ids_pool[i],
                        **self.metadata.get(ids_pool[i], {}),
                        "vertical": self.verticals.get(ids_pool[i]),
                        "similarity": float(similarities[i]),
                        "projection": coords_pool[i].astype(float).tolist(),
                    }
                    for i in farthest
                ],
            }

    def locate_profile(
        self,
        profile_info: dict,
        videos: list[dict],
        vectors: np.ndarray,
        neighbor_count: int = 12,
    ) -> dict:
        """Ubica un perfil/canal (hasta 40 videos) en el mapa 3D, mide su burbuja y recomienda fuera de ella."""
        with self.lock:
            vecs = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-12, None)

            centroid_vec = vecs.mean(axis=0)
            centroid_vec /= max(float(np.linalg.norm(centroid_vec)), 1e-12)
            cohesion = float((vecs @ centroid_vec).mean())

            prof_coords = np.zeros((len(videos), 3), dtype=np.float32)
            missing_idx = []
            for k, v in enumerate(videos):
                yid = str(v.get("youtube_id", ""))
                if yid in self.positions:
                    prof_coords[k] = self.coordinates[self.positions[yid]]
                else:
                    missing_idx.append(k)
            if missing_idx:
                prof_coords[missing_idx] = self.reducer.transform(vecs[missing_idx]).astype(np.float32)

            centroid_3d = prof_coords.mean(axis=0).astype(float).tolist()

            video_cats = []
            for k, v in enumerate(videos):
                yid = str(v.get("youtube_id", ""))
                cat = self.verticals.get(yid) or self._classify_vector_vertical(vecs[k]) or "(Sin categoría)"
                video_cats.append(cat)

            cat_counts = Counter(c for c in video_cats if c != "(Sin categoría)").most_common()
            total_cats_in_map = len(self.cat_names) or 25

            profile_ids = {str(v.get("youtube_id", "")) for v in videos}
            similarities = self.vectors @ centroid_vec
            near_sims = similarities.copy()
            for yid in profile_ids:
                if yid in self.positions:
                    near_sims[self.positions[yid]] = -np.inf
            nearest = np.argsort(-near_sims)[:neighbor_count]

            far_sims = similarities.copy()
            for yid in profile_ids:
                if yid in self.positions:
                    far_sims[self.positions[yid]] = np.inf
            far_pool = np.argsort(far_sims)[:400]
            farthest = []
            seen_verticals = set()
            for i in far_pool:
                v = self.verticals.get(self.ids[i])
                if v and v not in seen_verticals:
                    seen_verticals.add(v)
                    farthest.append(i)
                    if len(farthest) >= neighbor_count:
                        break
            if len(farthest) < neighbor_count:
                chosen = set(farthest)
                for i in far_pool:
                    if i not in chosen:
                        farthest.append(i)
                        if len(farthest) >= neighbor_count:
                            break

            knn_votes = {
                "vertical": cat_counts[0][0] if cat_counts else None,
                "count": cat_counts[0][1] if cat_counts else 0,
                "total": len(videos),
                "breakdown": [{"vertical": v, "count": c} for v, c in cat_counts],
            }

            return {
                "map_version": self.map_meta["version"],
                "profile": {
                    **profile_info,
                    "analyzed_videos": len(videos),
                    "categories_covered": len(cat_counts),
                    "total_categories": total_cats_in_map,
                    "cohesion": round(cohesion, 3),
                    "knn_votes": knn_votes,
                    "zero_shot_tags": self._predict_zero_shot(centroid_vec),
                },
                "projection": centroid_3d,
                "profile_videos": [
                    {
                        "youtube_id": v.get("youtube_id"),
                        "title": v.get("title"),
                        "vertical": video_cats[k],
                        "similarity": float(vecs[k] @ centroid_vec),
                        "projection": prof_coords[k].astype(float).tolist(),
                    }
                    for k, v in enumerate(videos)
                ],
                "neighbors": [
                    {
                        "youtube_id": self.ids[i],
                        **self.metadata.get(self.ids[i], {}),
                        "vertical": self.verticals.get(self.ids[i]),
                        "similarity": float(similarities[i]),
                        "projection": self.coordinates[i].astype(float).tolist(),
                    }
                    for i in nearest
                ],
                "farthest": [
                    {
                        "youtube_id": self.ids[i],
                        **self.metadata.get(self.ids[i], {}),
                        "vertical": self.verticals.get(self.ids[i]),
                        "similarity": float(similarities[i]),
                        "projection": self.coordinates[i].astype(float).tolist(),
                    }
                    for i in farthest
                ],
            }
