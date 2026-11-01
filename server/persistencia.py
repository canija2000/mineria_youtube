"""Guarda lo que llega por la web cuando el disco del contenedor se borra (Cloud Run).

En Cloud Run cada contenedor nace desde la imagen y lo que escribe se pierde al apagarse.
Si `MAPA_PERSISTENTE` apunta a una carpeta que sí persiste (un bucket de Cloud Storage
montado como carpeta), el servidor:

- al arrancar, junta el catálogo de videos recibidos (SQLite) de ahí con el que viene en
  la imagen y deja la unión en los dos lados. La primera vez lo siembra con el de la imagen;
- después de cada video nuevo, deja una copia del catálogo ahí;
- guarda ahí las miniaturas de los videos ingresados.

SQLite no se usa directo sobre el bucket porque esos montajes no manejan bien una base de
datos; se copia completo (pesa pocos MB). Con una sola instancia (`--max-instances 1`) no
hay dos copias escribiendo a la vez.

Sin `MAPA_PERSISTENTE` (en local) no hace nada y todo queda en `data/`.
"""
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from threading import Lock

from recolect.received import ReceivedStore, store_path

_lock = Lock()


def carpeta() -> Path | None:
    valor = os.getenv("MAPA_PERSISTENTE")
    return Path(valor) if valor else None


def carpeta_miniaturas(data_dir) -> Path:
    persistente = carpeta()
    return (persistente / "thumbs") if persistente else Path(data_dir) / "thumbs"


def restaurar(data_dir) -> None:
    """Al arrancar: junta el catálogo de la imagen con el del bucket y deja la unión en ambos.

    Se juntan en vez de reemplazar porque cada uno puede tener videos que el otro no: el bucket,
    los que llegaron por la web; la imagen, los agregados en local que entraron al mapa nuevo
    (si se perdieran, sus puntos desaparecerían del mapa aunque tengan coordenadas).
    """
    persistente = carpeta()
    if not persistente:
        return
    local, remoto = store_path(data_dir), persistente / store_path(data_dir).name
    persistente.mkdir(parents=True, exist_ok=True)
    if remoto.exists():
        with tempfile.TemporaryDirectory() as tmp:
            # se lee una copia en disco local: SQLite nunca abre el archivo del montaje
            copia = Path(tmp) / remoto.name
            shutil.copyfile(remoto, copia)
            nuevos = ReceivedStore(local).merge_from(copia)
        print(f"[persistencia] catálogo de videos recibidos restaurado desde {remoto} "
              f"(+{nuevos} que la imagen no tenía)", flush=True)
    else:
        print(f"[persistencia] primera vez: catálogo sembrado en {remoto}", flush=True)
    guardar(data_dir)


def guardar(data_dir) -> None:
    """Copia consistente del catálogo local a la carpeta persistente."""
    persistente = carpeta()
    if not persistente:
        return
    local = store_path(data_dir)
    remoto = persistente / local.name
    with _lock, tempfile.TemporaryDirectory() as tmp:
        # la API de backup de SQLite da una copia consistente aunque haya una escritura en curso;
        # se escribe en disco local y recién ahí se copia al bucket (SQLite nunca toca el montaje)
        copia = Path(tmp) / local.name
        with closing(sqlite3.connect(local)) as origen, closing(sqlite3.connect(copia)) as destino:
            origen.backup(destino)
        shutil.copyfile(copia, persistente / (local.name + ".tmp"))
        os.replace(persistente / (local.name + ".tmp"), remoto)
