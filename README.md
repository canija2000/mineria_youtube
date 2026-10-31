# Mapa de YouTube

Un mapa 3D explorable de ~35 mil videos de YouTube-8M, donde videos parecidos quedan cerca. Cada video se representa con EmbeddingGemma 2 a partir de su texto (título, tags, descripción) y de su miniatura, y la web permite ubicar en el mapa un video, un concepto, una imagen o un canal.

## Estructura del repo

```
recolect.py            # pipeline completo: YT-8M -> API de YouTube -> miniaturas -> embeddings -> mapas
remap.py               # recalcula el mapa 3D (texto o miniaturas) sin rehacer embeddings
recolect/              # módulos del pipeline (descarga, hidratación, embeddings, mapas, SQLite)
server/                # API FastAPI que sirve el mapa y ubica videos nuevos (+ la web en /)
web/index.html         # la web: visor Three.js conectado a la API
analysis/              # métricas y gráficos compartidos por los notebooks
notebooks/             # exploración guiada, en orden:
  01_datos                 qué videos hay y de dónde salen
  02_texto_vs_baselines    EmbeddingGemma frente a TF-IDF y MiniLM
  03_multimodal            texto, imagen, imagen + texto y fusión tardía
  04_zero_shot             etiquetar miniaturas con el vocabulario de YT-8M
  05_clustering_hdbscan    grupos del mapa y su bautizo zero-shot
informe/               # informe en LaTeX (main.tex) y sus figuras
docs/estado.md         # estado actual del proyecto y próximos pasos
docs/despliegue_oracle.md        # cómo publicar la web en una VM gratis de Oracle Cloud
docs/despliegue_huggingface.md   # cómo publicar la web en un Space (requiere HF PRO)
deploy/                # scripts de despliegue (oracle/: arranque y Caddy; Space de HF)
Dockerfile             # imagen del servidor + web (requirements-server.txt, versiones fijas)
data/                  # todo lo generado (no se versiona, salvo data/metricas.csv)
```

## Por dónde empezar

1. Prepara el entorno (sección siguiente) y consigue los datos: pide el paquete `mineria_youtube_datos.zip` (~650 MB) y descomprímelo en la raíz del repo con `unzip mineria_youtube_datos.zip`, o corre `python3 recolect.py` (varias horas).
2. Lee los notebooks en orden: solo leen lo que dejó el pipeline, no recalculan nada pesado.
3. Levanta el servidor (`uvicorn server.app:app --port 8000`) y abre `http://127.0.0.1:8000`. Con el paquete basta para explorar el mapa, buscar conceptos e imágenes y consultar los videos que ya están en el catálogo. Para agregar videos o perfiles nuevos hace falta `YOUTUBE_API_KEY` en `.env`. La primera vez se descarga EmbeddingGemma desde Hugging Face; si pide aceptar la licencia de Gemma, hay que aceptarla con tu cuenta y correr `huggingface-cli login`.

## Recolección

La preparación de datos completa se ejecuta con un solo comando:

```bash
python3 recolect.py
```

El script orquesta la descarga de YouTube-8M, extrae los pids, los traduce a IDs de YouTube, consulta metadatos, descarga miniaturas, recupera el vocabulario de YT-8M y calcula los espacios vectoriales. Es reanudable: reutiliza artefactos existentes y conserva las cachés de la API.

## Preparación

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv -r requirements.txt
echo 'YOUTUBE_API_KEY=tu_key' > .env
```

Activa el entorno antes de ejecutar el script. La llave de API se requiere para resolver los metadatos. El proyecto necesita espacio en disco para los shards YT-8M, miniaturas y archivos Parquet.

## Etapas y productos

1. Descarga por defecto el sample 1/100 de los shards `video/train` de YT-8M.
2. Extrae pids, etiquetas y features de YT-8M a `data/yt8m_sample.parquet`.
3. Resuelve `pid -> youtube_id` y lo guarda en `data/id_map.parquet`.
4. Consulta YouTube Data API v3 y guarda metadatos en `data/yt8m_hydrated.parquet`; `data/api_raw.jsonl` permite reanudar.
5. Descarga miniaturas a `data/thumbs/`.
6. Descarga el vocabulario a `data/yt8m_vocabulary.csv` y genera sus embeddings en `data/emb_vocab.parquet`.
7. Calcula embeddings de texto con EmbeddingGemma 2 y dos baselines comparables: TF-IDF+SVD y MiniLM.
8. Calcula embeddings de miniatura con EmbeddingGemma 2.
9. Calcula embeddings multimodales con imagen y texto en una sola entrada de EmbeddingGemma 2.
10. Calcula fusión tardía normalizando la suma del embedding de texto y el embedding de imagen.
11. Calcula y guarda el mapa 3D (UMAP) que usa el servidor (ver más abajo).

Los vectores por representación se guardan en `data/emb_text.parquet`, `data/emb_tfidf.parquet`, `data/emb_minilm.parquet`, `data/emb_image.parquet`, `data/emb_imgtext.parquet` y `data/emb_fusion.parquet`. La tabla `data/embeddings_consolidated.parquet` reúne metadatos, etiquetas y seis columnas de vectores, alineadas por pid; es la entrada para el análisis de métricas posterior. El manifiesto de salida es `data/recolect_manifest.json`.

Para una corrida reducida durante desarrollo (modo prueba):

```bash
python3 recolect.py --limit 200
```

El modo prueba escribe sus resultados en `data/dev/` y **nunca modifica `data/`**. Solo comparte con la corrida real las cachés que no dependen de cuántos videos se procesen: shards de YT-8M, `yt8m_sample.parquet`, `id_map.parquet`, `api_raw.jsonl`, miniaturas y vocabulario.

`--force` vuelve a calcular los embeddings existentes. La descarga de shards y las cachés de hidratación se mantienen. La extracción de frames y la búsqueda semántica frame por frame no forman parte de este pipeline.

Para solo completar las miniaturas que faltan, sin calcular vectores ni reescribir ningún archivo:

```bash
python3 recolect.py --solo-thumbnails
```

Para sumar videos actuales (populares por país y de canales semilla) al catálogo de la web, sin tocar la muestra de YT-8M:

```bash
python3 recolect.py --modern --regions CL,AR,MX,ES,US
```

Los embeddings de miniaturas corren sin barra de progreso. Mientras `recolect.py` está en esa etapa, el avance se consulta desde otra terminal:

```bash
python3 -m recolect.avance
```

## Métricas y notebooks

`python3 -m analysis.metricas` evalúa las seis representaciones sobre los mismos videos (kNN en el espacio original y en UMAP 3D, NMI de K-Means contra la categoría) y escribe `data/metricas.csv` y `data/proyecciones.parquet`, que leen los notebooks 02 y 03. Tarda ~5 minutos; solo hace falta volver a correrlo si cambian los embeddings.

Los notebooks guardan sus figuras en `informe/figs/` solo si se cambia `GUARDAR_FIGS = True` en su primera celda.

Para compartir los datos sin que el resto tenga que correr `recolect.py`, `python3 -m analysis.paquete` arma `mineria_youtube_datos.zip` (~650 MB). Trae lo que leen los notebooks y el servidor, con los embeddings en float16: los resultados no cambian (hit@k igual a 4 decimales). La mitad del peso es el molde UMAP del mapa de texto, que el servidor usa para ubicar videos nuevos. No incluye las miniaturas: los notebooks y el servidor bajan de YouTube las que necesitan. Cuando faltan `emb_text.parquet` y `emb_image.parquet`, el servidor toma los vectores del consolidado.

## Mapa 3D guardado y recálculo

El mapa (posiciones 3D y el "molde" UMAP para ubicar videos nuevos) se calcula **fuera del servidor** y se guarda por versiones:

```
data/maps/current.json        # cuál versión está vigente
data/maps/<versión>/coords.parquet, reducer.joblib, meta.json
data/maps/<versión>/clusters.parquet, clusters.json, centroids.npy   # grupos HDBSCAN con nombre zero-shot
data/maps_image/...           # lo mismo para el espacio de miniaturas
```

Junto a cada mapa se guardan sus grupos HDBSCAN (`recolect/clusters.py`, el mismo procedimiento del notebook 05). La web los usa para colorear el mapa por grupo en vez de por categoría de YouTube. Para agregarlos a un mapa ya calculado sin rehacer el UMAP: `python3 -m recolect.clusters --rep text` (o `--rep image`).

- `recolect.py` lo calcula en la etapa 11 si no existe o si cambiaron los vectores de texto (y el de miniaturas si cambió `emb_image.parquet`).
- Cada video que llega al servidor se guarda en `data/webapp_catalog.sqlite3` (metadatos + vector) y se ubica sobre el mapa vigente sin modificarlo.
- `remap.py` recalcula el mapa con los videos base **más los recibidos**, publica una versión nueva y conserva las 3 últimas (`--keep`). El servidor detecta la versión nueva en el siguiente pedido y la carga sin reiniciarse.

```bash
python3 remap.py                  # recalcula ahora
python3 remap.py --min-new 50     # solo si llegaron al menos 50 videos nuevos desde el último mapa
python3 remap.py --rep image      # mapa de miniaturas (solo videos base: SQLite no guarda vectores de imagen)
python3 remap.py --data data/dev  # sobre los datos de prueba
```

Para dejarlo como rutina, por ejemplo cada noche a las 3:00 con `crontab -e`:

```
0 3 * * * cd /ruta/a/mineria_youtube && .venv/bin/python remap.py --min-new 50 >> data/remap.log 2>&1
```

## Docker (servidor + web)

La imagen trae el código y las dependencias con versiones fijas (`requirements-server.txt`, torch solo-CPU), así el servidor corre igual en cualquier máquina. No incluye datos ni modelo:

- **Datos:** se montan en `/app/data`. Si la carpeta está vacía y está definida `DATA_URL` (un enlace de descarga directa al paquete de `python3 -m analysis.paquete`), `server/datos.py` lo descarga y descomprime al arrancar.
- **Modelo:** EmbeddingGemma se baja de Hugging Face la primera vez y queda en `/app/data/hf`. Si el modelo pide aceptar la licencia de Gemma, hay que pasar `HF_TOKEN` de una cuenta que la haya aceptado.
- **Persistencia:** con un volumen en `/app/data`, los videos que llegan por la web (SQLite) y sus miniaturas (`data/thumbs/`) sobreviven a los reinicios. Sin volumen se pierden al redeployar.

```bash
docker build -t mapa-youtube .
docker run -p 8000:8000 -v "$PWD/data:/app/data" --env-file .env mapa-youtube
```

Variables: `YOUTUBE_API_KEY` (para agregar videos y perfiles), `DATA_URL` (opcional), `HF_TOKEN` (si el modelo lo pide) y `PORT` (la definen Railway y otros; por defecto 8000). El servidor usa ~1,5–3 GB de RAM.

Para publicarla en una máquina gratis de Oracle Cloud (con Caddy y HTTPS), ver [docs/despliegue_oracle.md](docs/despliegue_oracle.md). La guía de Hugging Face Spaces ([docs/despliegue_huggingface.md](docs/despliegue_huggingface.md)) sigue disponible, pero los Spaces con Docker ahora requieren la suscripción PRO.

## API y web

El backend está en `server/app.py` y sirve también la web (`web/index.html`) en la raíz. Recibe un ID o URL de YouTube, consulta metadatos, genera el embedding, busca vecinos y ubica el nuevo punto sobre el mapa guardado.

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000
MAPA_DATA=data/dev uvicorn server.app:app --port 8000   # con los datos del modo prueba
```

La API usa `emb_text.parquet`, `emb_image.parquet`, `emb_vocab.parquet`, `yt8m_hydrated.parquet` y los mapas vigentes de `maps/` y `maps_image/` como catálogo base.

- `GET /map?rep=text|image` entrega, además de las categorías, el grupo HDBSCAN de cada punto (`cluster`, índice en la lista `clusters`). Entrega todos los puntos del mapa vigente (videos base + recibidos) en columnas: `youtube_id`, `x`, `y`, `z`, `title`, `channel_title`, `vertical` (índice en la lista `verticals`, -1 si no tiene), `source` (índice en `sources`: `yt8m` o `recibido`) e `in_map` (si el video ya formaba parte del cálculo del mapa o solo se ubicó sobre él). La respuesta va comprimida (gzip).
- `POST /videos` agrega o recupera un video y devuelve su posición, sus 12 vecinos, los 12 más lejanos y etiquetas zero-shot. Con `representation: "image"` lo ubica por su miniatura.
- `POST /concepts` ubica una palabra o frase libre y devuelve la región del mapa más afín.
- `POST /images` ubica una imagen cualquiera (URL o archivo arrastrado) en el espacio de miniaturas, sin guardarla.
- `POST /profiles` toma un `@handle`, canal o playlist, embebe sus últimos videos y mide qué tan concentrados están en el mapa.
- `GET /health` indica la versión del mapa y cuántos videos recibidos aún no están incluidos en él.

Los videos recibidos por `/videos` se guardan en `webapp_catalog.sqlite3` (metadatos y vector de texto) y su miniatura en `data/thumbs/`, sin importar si se buscaron en modo texto o imagen. Las imágenes sueltas de `/images` no se guardan. Hacer clic en un punto del mapa abre su panel sin guardar nada nuevo.
