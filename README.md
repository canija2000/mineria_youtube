## YouTube-8M + YouTube Data API v3

La idea: meter videos de YouTube en un espacio vectorial, ver cómo se agrupan por tema y (más adelante, ójala, usarlo para encontrar videos **fuera de tus recomendaciones**.)

Usamos dos fuentes:

1.  [**YouTube-8M**](https://research.google.com/youtube8m/) (Google, v2 2018): \~6,1M videos con features visuales/audio ya calculadas (`mean_rgb`, `mean_audio`) y etiquetas temáticas. Los IDs vienen anonimizados.
2.  [**YouTube Data API v3**](https://developers.google.com/youtube/v3): título, descripción, tags, vistas, miniatura, etc.

Del train de YT-8M bajamos un sample de **\~1%** (41 shards, 41.393 videos). Cada video trae un `pid` anonimizado, que traducimos al ID real de YouTube para pedir su metadata a la API. Con el texto (título + tags + descripción) generamos embeddings con [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) y los proyectamos en 3D.

```mermaid
flowchart TD
    A["YouTube-8M<br/>sample 1% (41k videos)"] -->|extract.py| B[("yt8m_sample.parquet<br/>pid · labels · mean_rgb · mean_audio")]
    B -->|"hydrate.py<br/>pid → youtube_id"| C["YouTube Data API v3"]
    C --> D[("yt8m_hydrated.parquet<br/>título · tags · descripción · stats")]
    D -->|"embed.py<br/>EmbeddingGemma 2"| E[("emb_text.parquet<br/>768 dims")]
    E -->|review.ipynb| F["PCA · UMAP 3D"]
    B -.->|"labels → categoría"| F
    F --> G[("projections.parquet")]
    G -->|preview.ipynb| H["Visualización 3D<br/>+ métricas"]
```

### Los archivos

- `extract.py`: lee los `.tfrecord` de YT-8M y los pasa a `data/yt8m_sample.parquet`.
- `hydrate.py`: traduce `pid` → `youtube_id` y pide la metadata a la API → `data/yt8m_hydrated.parquet`.
- `embed.py`: arma el texto de cada video y lo embebe con EmbeddingGemma 2 → `data/emb_text.parquet`.
- `review.ipynb`: el paso a paso: exploración, PCA, UMAP, métricas → `data/projections.parquet`.
- `preview.ipynb`: para ver los resultados directo, sin recalcular nada.

### El embudo de datos

| Etapa                    | Videos             |
| ------------------------ | ------------------ |
| Sample YT-8M             | 41.393             |
| Con ID real de YouTube   | 37.230 (89,9%)     |
| Todavía vivos en YouTube | **34.788 (84,0%)** |

### Cómo correrlo

```bash
# 1. entorno
uv venv --python 3.12 .venv && uv pip install -p .venv -r requirements.txt

# 2. API key de YouTube (Google Cloud → habilitar "YouTube Data API v3" → crear API key)
echo 'YOUTUBE_API_KEY=tu_key' > .env

# 3. bajar el sample (1/100 del train)
mkdir -p data/yt8m/video/train && cd data/yt8m/video/train
curl data.yt8m.org/download.py | shard=1,100 partition=2/video/train mirror=us python3
cd -

# 4. pipeline
.venv/bin/python extract.py
.venv/bin/python hydrate.py     # ~10 min, ~750 unidades de cuota
.venv/bin/python embed.py       # ~30 min en un M2
```

### implementado

1.  Se agregan modelos _base-lines_ para comparar con EmbeddingGemma 2, y se guardan sus métricas en `data/metricas_modelos.csv` (trustworthiness, kNN acc).
    - Base Line 1 : `TF-IDF` : corresponde a un embedding clásico de texto que representa la frecuencia de las palabras en el corpus. Se hace un SVD para reducir la dimensionalidad a 768 dimensiones, para que sea comparable con EmbeddingGemma 2.
    - Base Line 2 : `BERT` : corresponde a un embedding de texto basado [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) que genera representaciones vectoriales de los textos. Es más avanzado que el anterior, pero, en teoría, menos que Gemma2. También se utiliza como marco comparativo. La gran diferencia con Gemma2, es que el espacio es de 384 dimensiones en lugar de 768.
2.  Se observan diferencias en las métricas de los modelos.
3.  - En `preview_images.ipynb` se trabajan con los resultados consolidados del proceso de `multimodal.py` .
    - `multimodal.py` hace lo siguiente \[muestra de 5000 videos\]:
      1.  embedding de la imagen $\rightarrow$ `emb_image.parquet`
      2.  concatenación de `text` + `image` como input a un solo vector $\rightarrow$ `emb_imgtext.parquet`
      3.  embedding de $3806$ etiquitas de YT-8M $\rightarrow$ `emb_vocab.parquet`
      4.  zero-shot \[top-10 etiquetas x imagene\] $\rightarrow$ `zeroshot_image.parquet` Las imagenes se obtienen a partir del `youtube_id` \[obtenidos en el proceso anterior\] , a partir de este se solicita el thumbnail del video `https://i.ytimg.com/vi/{youtube_id}/mqdefault.jpg` .
    - Sobre el _zero-shot_ : dado que el embedding de la imagen y el embedding de las etiquetas están en el mismo espacio vectorial, se puede calcular la similitud entre ellos \[multiplicando las matrices\] y obtener las etiquetas más cercanas a cada imagen. Esto permite etiquetar imágenes sin necesidad de un modelo entrenado específicamente para ello.
