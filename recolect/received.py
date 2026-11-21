"""Videos recibidos por el servidor.

Cada link que llega a la webapp se guarda aquí con sus metadatos y su vector de texto.
La tabla `space_vectors` guarda además los vectores de miniatura ("image") y de fusión temprana
("imgtext") que el servidor calcula al consultar un video en esos mapas, para no volver a
embeber la imagen (la fusión tardía sale de texto + imagen y no se guarda).
El servidor los ubica sobre el mapa vigente, y `remap.py` los incluye la próxima
vez que recalcula el mapa, para que el mapa no quede fijo en los videos viejos de YT-8M.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_NAME = "webapp_catalog.sqlite3"


def store_path(data_dir="data") -> Path:
    return Path(data_dir) / DEFAULT_NAME


class ReceivedStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS received (
                youtube_id TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                embedding BLOB NOT NULL,
                received_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS space_vectors (
                youtube_id TEXT NOT NULL,
                space TEXT NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (youtube_id, space)
            )""")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def add(self, youtube_id: str, metadata: dict, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        received_at = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR IGNORE INTO received VALUES (?, ?, ?, ?)",
                       (youtube_id, json.dumps(metadata, ensure_ascii=False), vector.tobytes(), received_at))

    def add_many(self, items: list[tuple[str, dict, np.ndarray]]) -> int:
        """Inserta un lote de videos (youtube_id, metadata, vector) en una sola transacción."""
        if not items:
            return 0
        received_at = datetime.now(timezone.utc).isoformat()
        payload = [
            (
                str(yid),
                json.dumps(meta, ensure_ascii=False),
                np.asarray(vec, dtype=np.float32).reshape(-1).tobytes(),
                received_at,
            )
            for yid, meta, vec in items
        ]
        with closing(self._connect()) as db, db:
            before = db.execute("SELECT COUNT(*) FROM received").fetchone()[0]
            db.executemany("INSERT OR IGNORE INTO received VALUES (?, ?, ?, ?)", payload)
            after = db.execute("SELECT COUNT(*) FROM received").fetchone()[0]
        return int(after - before)

    def all(self) -> list[dict]:
        """Todos los videos recibidos, del más antiguo al más nuevo."""
        with closing(self._connect()) as db:
            rows = db.execute("SELECT youtube_id, metadata_json, embedding, received_at "
                              "FROM received ORDER BY received_at").fetchall()
        return [{"youtube_id": video_id, "metadata": json.loads(metadata_json),
                 "vector": np.frombuffer(blob, dtype=np.float32).copy(), "received_at": received_at}
                for video_id, metadata_json, blob, received_at in rows]

    def get(self, youtube_id: str) -> dict | None:
        """Metadatos completos de un video recibido (con tags y descripción), o None si no está."""
        with closing(self._connect()) as db:
            row = db.execute("SELECT metadata_json FROM received WHERE youtube_id = ?", (youtube_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def add_space_vectors(self, space: str, items: dict[str, np.ndarray]) -> None:
        """Guarda vectores de un espacio ("image" o "imgtext") por youtube_id; no pisa los que ya están."""
        if not items:
            return
        payload = [(str(yid), space, np.asarray(vec, dtype=np.float32).reshape(-1).tobytes())
                   for yid, vec in items.items()]
        with closing(self._connect()) as db, db:
            db.executemany("INSERT OR IGNORE INTO space_vectors VALUES (?, ?, ?)", payload)

    def space_vectors(self, space: str, youtube_ids=None) -> dict[str, np.ndarray]:
        """Vectores guardados de un espacio: de esos ids, o de todos si `youtube_ids` es None."""
        with closing(self._connect()) as db:
            if youtube_ids is None:
                rows = db.execute("SELECT youtube_id, embedding FROM space_vectors WHERE space = ?", (space,)).fetchall()
            else:
                ids = list(youtube_ids)
                rows = []
                for i in range(0, len(ids), 500):
                    lote = ids[i:i + 500]
                    rows += db.execute(f"SELECT youtube_id, embedding FROM space_vectors WHERE space = ? "
                                       f"AND youtube_id IN ({','.join('?' * len(lote))})", (space, *lote)).fetchall()
        return {yid: np.frombuffer(blob, dtype=np.float32).copy() for yid, blob in rows}

    def count(self) -> int:
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM received").fetchone()[0]

    def merge_from(self, other_path) -> int:
        """Agrega los videos de otro catálogo que aquí no están (nunca borra ni pisa). Devuelve cuántos entraron.

        Sirve para juntar el catálogo de Cloud Run (bucket) con el local antes de recalcular el mapa.
        """
        with closing(self._connect()) as db, db:
            before = db.execute("SELECT COUNT(*) FROM received").fetchone()[0]
            db.execute("ATTACH DATABASE ? AS otro", (str(other_path),))
            db.execute("INSERT OR IGNORE INTO received SELECT youtube_id, metadata_json, embedding, received_at "
                       "FROM otro.received")
            # catálogos anteriores a esta tabla no la tienen
            if db.execute("SELECT 1 FROM otro.sqlite_master WHERE name = 'space_vectors'").fetchone():
                db.execute("INSERT OR IGNORE INTO space_vectors SELECT youtube_id, space, embedding "
                           "FROM otro.space_vectors")
            after = db.execute("SELECT COUNT(*) FROM received").fetchone()[0]
        return int(after - before)


if __name__ == "__main__":
    # python3 -m recolect.received <otro.sqlite3> [carpeta-de-datos]: suma al catálogo local los videos de otro
    import sys

    otro, data_dir = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "data")
    store = ReceivedStore(store_path(data_dir))
    nuevos = store.merge_from(otro)
    print(f"[catálogo] {nuevos} videos nuevos desde {otro}; {store.count():,} recibidos en {store.path}")
