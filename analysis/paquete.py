"""Paquete liviano de datos para correr los notebooks sin `recolect.py`.

`recolect.py` tarda horas (embeddings de 35 mil textos y miniaturas). Los notebooks
solo leen lo que dejó, así que basta con compartir esos archivos:

  data/embeddings_consolidated.parquet   las 6 representaciones, en float16 (580 MB -> ~285 MB)
  data/emb_vocab.parquet                 vocabulario YT-8M embebido
  data/proyecciones.parquet              UMAP 3D de cada representación (notebooks 02 y 03)
  data/yt8m_hydrated.parquet             metadatos de la API (notebook 01)
  data/modern_videos.parquet             videos actuales de `--modern` (notebook 01)
  data/webapp_catalog.sqlite3            videos recibidos por la web (notebook 01)
  data/maps/current.json + coords        posiciones del mapa vigente, sin el molde UMAP (notebook 05)

float16 basta: el error por coordenada es < 0,0002, hit@k del zero-shot da igual a 4
decimales y los 5 vecinos más cercanos coinciden en más del 99 % de los casos.
Las miniaturas no van: los notebooks bajan de YouTube las pocas que muestran.

Uso:
    python3 -m analysis.paquete            # escribe mineria_youtube_datos.zip en la raíz del repo

Quien lo recibe, desde la raíz del repo:
    unzip mineria_youtube_datos.zip        # crea data/ con lo necesario
"""
import json
import tempfile
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .metricas import DATA, ROOT

SALIDA = ROOT / "mineria_youtube_datos.zip"
TAL_CUAL = ["emb_vocab.parquet", "proyecciones.parquet", "yt8m_hydrated.parquet",
            "modern_videos.parquet", "webapp_catalog.sqlite3"]


def consolidado_float16(destino: Path) -> None:
    """Copia de embeddings_consolidated.parquet con los vectores en float16 y compresión zstd."""
    tabla = pq.read_table(DATA / "embeddings_consolidated.parquet")
    columnas = {
        nombre: (tabla[nombre].cast(pa.list_(pa.float16())) if nombre.startswith("emb_") else tabla[nombre])
        for nombre in tabla.column_names
    }
    pq.write_table(pa.table(columnas), destino, compression="zstd", compression_level=19)


def main():
    version = json.loads((DATA / "maps" / "current.json").read_text())["version"]
    mapa = DATA / "maps" / version
    archivos = [mapa / nombre for nombre in ("coords.parquet", "meta.json", "clusters.parquet", "clusters.json")
                if (mapa / nombre).exists()]

    with tempfile.TemporaryDirectory() as tmp:
        consolidado = Path(tmp) / "embeddings_consolidated.parquet"
        print("convirtiendo embeddings a float16 (~1 min)...", flush=True)
        consolidado_float16(consolidado)

        # ZIP_STORED: los parquet ya vienen comprimidos, volver a comprimirlos no ahorra nada
        with zipfile.ZipFile(SALIDA, "w", compression=zipfile.ZIP_STORED) as z:
            z.write(consolidado, "data/embeddings_consolidated.parquet")
            for nombre in TAL_CUAL:
                z.write(DATA / nombre, f"data/{nombre}")
            z.write(DATA / "maps" / "current.json", "data/maps/current.json")
            for archivo in archivos:
                z.write(archivo, f"data/maps/{version}/{archivo.name}")

    print(f"-> {SALIDA.name} ({SALIDA.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
