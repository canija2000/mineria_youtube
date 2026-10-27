# Recolección y embeddings de videos de YouTube

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

## Mapa 3D guardado y recálculo

El mapa (posiciones 3D y el "molde" UMAP para ubicar videos nuevos) se calcula **fuera del servidor** y se guarda por versiones:

```
data/maps/current.json        # cuál versión está vigente
data/maps/<versión>/coords.parquet, reducer.joblib, meta.json
```

- `recolect.py` lo calcula en la etapa 11 si no existe o si cambiaron los vectores de texto.
- Cada video que llega al servidor se guarda en `data/webapp_catalog.sqlite3` (metadatos + vector) y se ubica sobre el mapa vigente sin modificarlo.
- `remap.py` recalcula el mapa con los videos base **más los recibidos**, publica una versión nueva y conserva las 3 últimas (`--keep`). El servidor detecta la versión nueva en el siguiente pedido y la carga sin reiniciarse.

```bash
python3 remap.py                  # recalcula ahora
python3 remap.py --min-new 50     # solo si llegaron al menos 50 videos nuevos desde el último mapa
python3 remap.py --data data/dev  # sobre los datos de prueba
```

Para dejarlo como rutina, por ejemplo cada noche a las 3:00 con `crontab -e`:

```
0 3 * * * cd /ruta/a/mineria_youtube && .venv/bin/python remap.py --min-new 50 >> data/remap.log 2>&1
```

## API para la webapp

El backend inicial está en `server/app.py`. Recibe un ID o URL de YouTube, consulta metadatos, genera el embedding de texto, busca vecinos y ubica el nuevo punto sobre el mapa guardado.

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000
MAPA_DATA=data/dev uvicorn server.app:app --port 8000   # con los datos del modo prueba
```

La API usa `emb_text.parquet`, `yt8m_hydrated.parquet` y el mapa vigente de `maps/` como catálogo base.

- `GET /map` entrega todos los puntos del mapa vigente (videos base + recibidos) en columnas: `youtube_id`, `x`, `y`, `z`, `title`, `channel_title`, `vertical` (índice en la lista `verticals`, -1 si no tiene), `source` (índice en `sources`: `yt8m` o `recibido`) e `in_map` (si el video ya formaba parte del cálculo del mapa o solo se ubicó sobre él). La respuesta va comprimida (gzip).
- `POST /videos` agrega o recupera un video y devuelve su posición y sus 12 vecinos.
- `GET /health` indica la versión del mapa y cuántos videos recibidos aún no están incluidos en él.

Los videos recibidos se guardan en `webapp_catalog.sqlite3`.
