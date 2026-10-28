"""Videos recibidos por el servidor.

Cada link que llega a la webapp se guarda aquí con sus metadatos y su vector.
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

    def count(self) -> int:
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM received").fetchone()[0]
