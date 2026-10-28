"""Avance de los embeddings de miniaturas mientras corre `recolect.py`.

`SentenceTransformer.encode()` corre sin barra de progreso, pero cada lote abre
sus miniaturas desde `data/thumbs/`, y eso actualiza su fecha de último acceso
(`st_atime`). Contar cuántas miniaturas se leyeron desde que empezó la fase
actual da el avance, sin importar en qué orden las recorra el modelo.

Fases (ver `recolect/modalities.py::embed_images`):
  1. imagen sola     -> escribe data/emb_image.parquet al terminar
  2. imagen + texto  -> escribe data/emb_imgtext.parquet al terminar

Uso:
    python3 -m recolect.avance
"""
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
THUMBS = DATA / "thumbs"


def _recolect_start() -> float | None:
    """Hora de inicio del proceso `recolect.py` en curso, o None si no corre."""
    pids = subprocess.run(["pgrep", "-f", "recolect.py"], capture_output=True, text=True).stdout.split()
    if not pids:
        return None
    lstart = subprocess.run(["ps", "-o", "lstart=", "-p", pids[0]], capture_output=True, text=True).stdout.strip()
    return datetime.strptime(lstart, "%a %b %d %H:%M:%S %Y").timestamp()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def main():
    start = _recolect_start()
    if start is None:
        print("recolect.py no está corriendo.")
        for name in ("emb_image", "emb_imgtext", "emb_fusion", "embeddings_consolidated"):
            path = DATA / f"{name}.parquet"
            print(f"  {name:<24} {time.ctime(_mtime(path)) if path.exists() else '—'}")
        return

    # Si emb_image.parquet se escribió durante esta corrida, ya vamos en imagen + texto.
    image_done = _mtime(DATA / "emb_image.parquet")
    if image_done > start:
        phase, since = "imagen + texto", image_done
    else:
        phase, since = "imagen", start

    total = read = 0
    for entry in os.scandir(THUMBS):
        if entry.name.endswith(".jpg"):
            total += 1
            read += entry.stat().st_atime > since

    elapsed_min = (time.time() - since) / 60
    rate = read / elapsed_min if elapsed_min > 0 else 0
    left = total - read
    eta = f"~{left / rate / 60:.1f} h" if rate else "?"
    print(f"Fase {phase}: {read:,}/{total:,} ({read / total * 100:.1f} %)")
    print(f"Velocidad: {rate:.0f} miniaturas/min · faltan {left:,} · {eta}")


if __name__ == "__main__":
    main()
