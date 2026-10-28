# Estado del proyecto: Mapa de YouTube

**Actualizado:** 27 de octubre de 2026

## 1. Resumen

- **Datos:** de una muestra de 41.393 videos de YouTube-8M, 34.788 siguen públicos con metadatos y 34.771 tienen miniatura. A eso se suman 1.243 videos actuales (`recolect.py --modern`) y los que se consultan desde la web, guardados en `data/webapp_catalog.sqlite3`.
- **Representaciones:** seis espacios para los mismos videos: TF-IDF, MiniLM y cuatro de EmbeddingGemma 2 (texto, imagen, imagen + texto en una pasada y fusión tardía). Todos completos para los 34.771 videos con miniatura.
- **Mapas:** UMAP 3D versionado para texto (`data/maps/`) y para miniaturas (`data/maps_image/`). Se calculan fuera del servidor con `recolect.py` o `remap.py`.
- **Servidor y web:** FastAPI sirve el mapa y la web en `http://127.0.0.1:8000`. Ubica videos, conceptos, imágenes y canales, con vecinos, lejanos y etiquetas zero-shot. La web ya no inventa resultados: si el servidor falla, muestra el error.
- **Análisis:** `analysis/` + `notebooks/01–05` reemplazan a los notebooks exploratorios anteriores.

## 2. Resultados principales

Sobre los 34.771 videos con todas las representaciones (`data/metricas.csv`, notebooks 02 y 03):

| representación | kNN (768-d) | kNN (UMAP 3D) | NMI K-Means (UMAP 3D) |
|---|---:|---:|---:|
| TF-IDF | 0,609 | 0,560 | 0,134 |
| MiniLM | 0,698 | 0,675 | 0,297 |
| Gemma · texto | 0,798 | 0,771 | 0,378 |
| Gemma · imagen | 0,784 | 0,769 | 0,391 |
| Gemma · imagen + texto | 0,819 | 0,791 | 0,364 |
| **Gemma · fusión tardía** | **0,838** | **0,823** | **0,452** |
| baseline de mayoría | 0,347 | | |

- EmbeddingGemma supera a los dos baselines de texto en las tres métricas.
- La miniatura sola queda casi a la par del texto, y la fusión tardía es la mejor en todo.
- UMAP conserva bien los vecinos (trustworthiness 0,956 frente a 0,828 de PCA 3D, notebook 02).
- Zero-shot con el vocabulario de YT-8M (notebook 04): desde la miniatura sola, al menos una etiqueta real está entre las 10 predichas en el 70 % de los videos (baseline: 56 %); con la fusión tardía, en el 90 %.

## 3. Componentes

| componente | qué hace | archivo principal |
|---|---|---|
| Pipeline | YT-8M → API de YouTube → miniaturas → embeddings → mapas | `recolect.py`, `recolect/pipeline.py` |
| Avance de miniaturas | progreso de los embeddings de imagen mientras corre el pipeline | `python3 -m recolect.avance` |
| Mapas versionados | UMAP 3D de texto y de miniaturas + molde para ubicar videos nuevos | `remap.py`, `recolect/mapping.py` |
| Videos recibidos | SQLite con metadatos y vector de texto | `recolect/received.py` |
| API | `/health`, `/map`, `/videos`, `/concepts`, `/images`, `/profiles` | `server/app.py`, `server/catalog.py` |
| Web | visor Three.js conectado a la API | `web/index.html` |
| Métricas | evaluación común de las seis representaciones | `analysis/metricas.py` |
| Notebooks | exploración guiada 01–05 | `notebooks/` |

## 4. Próximos pasos

1. **Fusión tardía en el servidor.** Es la mejor representación medida, pero el mapa de texto y los videos recibidos siguen usando solo el vector de texto. Habría que guardar también el vector de imagen de cada video recibido y publicar un mapa de fusión.
2. **Videos recibidos en el mapa de miniaturas.** SQLite no guarda vectores de imagen, así que esos videos se ubican al consultarlos pero no entran en `remap.py --rep image`.
3. **Línea de tiempo por frames.** Idea acordada, sin implementar: usar los storyboards de YouTube (intervalo proporcional al largo del video) y solo para los videos que ingresa el usuario. Cada frame se compara con el vocabulario igual que una miniatura.
4. **Perfil por likes.** La API solo entrega los likes de la cuenta propia y con OAuth (`videos.list(myRating=like)`). Hoy `/profiles` usa los últimos videos públicos de un canal o playlist.
5. **Actualizar el informe.** Las figuras de `informe/figs/` son de la muestra anterior de 5.000 videos. Se regeneran corriendo los notebooks 02–04 con `GUARDAR_FIGS = True`, y luego hay que revisar las cifras del texto.
