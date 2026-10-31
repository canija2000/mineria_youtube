"""Descarga el paquete de datos al arrancar, si `data/` está vacío (pensado para Docker / Railway).

El paquete es el zip de `analysis/paquete.py`. Se baja desde `DATA_URL` (cualquier enlace de
descarga directa) y se descomprime en `MAPA_DATA` (por defecto `data/`). Si ya hay un mapa,
no hace nada, así que en un volumen persistente solo descarga la primera vez.

Uso:
    DATA_URL=https://... python3 -m server.datos
"""
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import requests


def main():
    data = Path(os.getenv("MAPA_DATA", "data"))
    if (data / "maps" / "current.json").exists():
        print(f"[datos] {data}/ ya tiene un mapa; no se descarga nada", flush=True)
        return
    url = os.getenv("DATA_URL")
    if not url:
        sys.exit(f"[datos] {data}/ está vacío y no hay DATA_URL: monta los datos o define DATA_URL "
                 "con un enlace al paquete (python3 -m analysis.paquete)")

    data.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=data) as tmp:
        print(f"[datos] descargando el paquete en {data}/ ...", flush=True)
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
        tmp.flush()
        if not zipfile.is_zipfile(tmp.name):
            sys.exit("[datos] lo descargado no es un zip: revisa que DATA_URL sea un enlace de descarga directa")
        with zipfile.ZipFile(tmp.name) as z:
            # las rutas del zip empiezan con data/; se descomprimen dentro de MAPA_DATA
            for member in z.infolist():
                if member.is_dir() or not member.filename.startswith("data/"):
                    continue
                target = data / member.filename.removeprefix("data/")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
    print(f"[datos] listo: {sum(1 for _ in data.rglob('*') if _.is_file())} archivos en {data}/", flush=True)


if __name__ == "__main__":
    main()
