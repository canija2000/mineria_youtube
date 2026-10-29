"""Paquete de datos para correr los notebooks y el servidor sin `recolect.py`.

`recolect.py` tarda horas (embeddings de 35 mil textos y miniaturas). Los notebooks
y el servidor solo leen lo que dejó, así que basta con compartir esos archivos:

  data/embeddings_consolidated.parquet   las 6 representaciones, en float16 (580 MB -> ~285 MB)
  data/emb_vocab.parquet                 vocabulario YT-8M embebido
  data/proyecciones.parquet              UMAP 3D de cada representación (notebooks 02 y 03)
  data/yt8m_hydrated.parquet             metadatos de la API (notebook 01)
  data/modern_videos.parquet             videos actuales de `--modern` (notebook 01)
  data/webapp_catalog.sqlite3            videos recibidos por la web (notebook 01)
  data/maps/...                          mapa de texto vigente con su molde UMAP (notebook 05 y servidor)
  data/maps_image/...                    mapa de miniaturas, sin su molde UMAP (el servidor no lo usa)

El servidor saca los vectores de texto e imagen del consolidado cuando no encuentra
emb_text.parquet / emb_image.parquet (ver `recolect/mapping.py::load_base_vectors`).

float16 basta: el error por coordenada es < 0,0002, hit@k del zero-shot da igual a 4
decimales y los 5 vecinos más cercanos coinciden en más del 99 % de los casos.
Las miniaturas no van: los notebooks bajan de YouTube las pocas que muestran, y el
servidor baja la de cada video que se consulta.

Uso:
    python3 -m analysis.paquete            # escribe mineria_youtube_datos.zip en la raíz del repo

Quien lo recibe, desde la raíz del repo:
    unzip mineria_youtube_datos.zip        # crea data/ con lo necesario
    uvicorn server.app:app --port 8000     # opcional: la web en http://127.0.0.1:8000
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


def archivos_de_mapa(carpeta: str, con_molde: bool) -> list[Path]:
    """current.json y los archivos de la versión vigente de data/<carpeta>/."""
    version = json.loads((DATA / carpeta / "current.json").read_text())["version"]
    nombres = ["coords.parquet", "meta.json", "clusters.parquet", "clusters.json", "centroids.npy"]
    if con_molde:
        nombres.append("reducer.joblib")
    return [DATA / carpeta / "current.json"] + [
        DATA / carpeta / version / n for n in nombres if (DATA / carpeta / version / n).exists()]


def main():
    archivos = archivos_de_mapa("maps", con_molde=True) + archivos_de_mapa("maps_image", con_molde=False)

    with tempfile.TemporaryDirectory() as tmp:
        consolidado = Path(tmp) / "embeddings_consolidated.parquet"
        print("convirtiendo embeddings a float16 (~1 min)...", flush=True)
        consolidado_float16(consolidado)

        # ZIP_STORED: los parquet ya vienen comprimidos, volver a comprimirlos no ahorra nada
        with zipfile.ZipFile(SALIDA, "w", compression=zipfile.ZIP_STORED) as z:
            z.write(consolidado, "data/embeddings_consolidated.parquet")
            for nombre in TAL_CUAL:
                z.write(DATA / nombre, f"data/{nombre}")
            for archivo in archivos:
                # el molde UMAP sí se comprime un poco (374 -> ~330 MB)
                tipo = zipfile.ZIP_DEFLATED if archivo.suffix == ".joblib" else zipfile.ZIP_STORED
                z.write(archivo, f"data/{archivo.relative_to(DATA)}", compress_type=tipo)

    print(f"-> {SALIDA.name} ({SALIDA.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
