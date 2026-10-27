# Informe de estado: Mapa de YouTube (Actualizado)

**Fecha de actualización:** 27 de octubre de 2026  
**Alcance:** Flujo interactivo **"pego un link de video → aparece en el mapa con sus vecinos y metadatos completos"**.  
*En pausa deliberada según roadmap:* análisis por frames y perfil de usuario (40 likes).

---

## 1. Resumen ejecutivo

El proyecto ya cuenta con un pipeline estructurado y un sistema cliente-servidor funcionando en vivo:
1. **Pipeline y datos protegidos:** El comando de recolección (`recolect.py`) y el modo de pruebas (`--limit N`) ahora aíslan sus salidas en `data/dev/`, garantizando que nunca se sobreescriban los 34.788 videos reales de `data/`.
2. **Mapa UMAP persistente y versionado:** El mapa 3D ya no se recalcula al encender el servidor; se genera y guarda por versiones en `data/maps/` mediante `remap.py`. Los nuevos videos recibidos se almacenan en SQLite (`webapp_catalog.sqlite3`) para alimentar futuros recálculos del mapa.
3. **Servidor FastAPI activo:** Expone `GET /map` (entrega los 34.788 puntos comprimidos con Gzip en ~0.5s y ~1.98 MB), `POST /videos` (ubica videos, extrae metadatos y calcula sus 12 vecinos más cercanos) y sirve la webapp en su raíz (`http://127.0.0.1:8000`).
4. **Frontend 3D interactivo y conectado:** El frontend en Three.js ahora consume la API en vivo, permanece estático para navegación cómoda con el mouse (sin auto-rotación automática molesta) y cuenta con un panel lateral derecho enriquecido con enlaces a YouTube, vistas, likes, duración, canal y botones para centrar cada vecino en el mapa.

---

## 2. Estado de resolución de problemas previos

| # | Problema detectado inicialmente | Estado | Solución implementada |
|---|---|:---:|---|
| **1** | El servidor recalculaba el mapa UMAP en cada arranque (lento e inestable). | ✅ **Resuelto** | Se creó el módulo `recolect/mapping.py` y la rutina `remap.py`. El mapa se calcula una sola vez y se almacena en `data/maps/<versión>/`. El servidor solo lo lee en memoria. |
| **2** | El modo prueba (`--limit`) sobreescribía los datos reales en `data/`. | ✅ **Resuelto** | El pipeline (`recolect/pipeline.py`) ahora redirige automáticamente cualquier ejecución con `--limit` a `data/dev/`. |
| **3** | Riesgo de recalcular vectores completos al correr de nuevo. | ✅ **Resuelto** | Se incorporó detección inteligente de artefactos existentes y la bandera `--solo-thumbnails` para descargar únicamente miniaturas pendientes sin recalcular embeddings. |
| **4** | La web no tenía cómo pedir el mapa base de 34 mil puntos. | ✅ **Resuelto** | Endpoint `GET /map` creado con compresión `GZipMiddleware` y estructura columnar compacta, incluyendo categorías temáticas (*verticals*). |
| **5** | El frontend era solo una maqueta con datos simulados y 404 en raíz. | ✅ **Resuelto** | FastAPI ahora monta `webapp_mockup/` en `/`. `index.html` se conectó a `GET /map` y `POST /videos`. |
| **6** | El mapa 3D rotaba solo continuamente impidiendo inspeccionar. | ✅ **Resuelto** | Desactivada la propiedad `autoRotate`. El mapa ahora permanece estático y se explora con el mouse. |
| **7** | Al ingresar un video solo salían títulos sin links a YouTube. | ✅ **Resuelto** | Se expandió `_card` en `server/catalog.py` para devolver vistas, likes, duración, fecha y descripción. En el front se añadieron enlaces directos `target="_blank"` a YouTube para el video y sus vecinos, más un botón de centrado 3D. |

---

## 3. Matriz de componentes y estado actual

| Componente | Qué hace | Estado | Archivo principal |
|---|---|:---:|---|
| **Base de datos base** | 34.788 videos vivos de YouTube-8M con texto y categorías | ✅ Listo | `data/yt8m_hydrated.parquet` |
| **Vectores de texto** | Embeddings 768-d con EmbeddingGemma 2 | ✅ Listo | `data/emb_text.parquet` |
| **Vectores baselines** | Comparativas de TF-IDF + SVD y MiniLM | ✅ Listo | `data/emb_tfidf.parquet`, `emb_minilm.parquet` |
| **Miniaturas descargadas** | Portadas descargadas para modalidades visuales | 🟡 5.000 listas | `data/thumbs/` (`recolect.py --solo-thumbnails`) |
| **Vectores imagen / multimodal** | Embeddings de miniatura y fusión tardía | 🟡 5.000 listos | `data/emb_image.parquet`, `emb_imgtext.parquet` |
| **Mapa UMAP 3D persistente** | Coordenadas 3D de los 34.788 videos y reductor entrenado | ✅ Listo | `data/maps/20261027-163059/` |
| **Rutina de recálculo** | Integra videos nuevos del servidor al mapa base | ✅ Listo | `remap.py` (`recolect/mapping.py`) |
| **Almacén de videos nuevos** | SQLite para persistir links agregados por usuarios | ✅ Listo | `data/webapp_catalog.sqlite3` (`recolect/received.py`) |
| **API Backend** | Servidor FastAPI (`/health`, `/map`, `/videos`) | ✅ Activo | `server/app.py`, `server/catalog.py` |
| **Frontend Web 3D** | Visualizador Three.js conectado a la API en vivo | ✅ Activo | `webapp_mockup/index.html` |
| **Modo seguro de desarrollo** | Prueba reducida sin tocar datos reales | ✅ Listo | `python3 recolect.py --limit N` (`data/dev/`) |

---

## 4. Próximos pasos pendientes (Roadmap)

1. **Fusión tardía para el video nuevo en el servidor:**
   - Actualmente el servidor ubica el video consultado usando el vector de texto (EmbeddingGemma 2).
   - Según el informe, la representación óptima es la fusión tardía (texto + imagen). Cuando se complete la recolección de imágenes de la base, el servidor podrá generar también el embedding visual de la miniatura del video nuevo y proyectar la fusión.
2. **Palabras zero-shot de la miniatura en el servidor:**
   - Usar `data/emb_vocab.parquet` para predecir las etiquetas más probables de la miniatura al recibir un video nuevo y mostrarlas en el panel como chips.
3. **Consagración completa de miniaturas (opcional en segundo plano):**
   - Ejecutar `python3 recolect.py --solo-thumbnails` para descargar las ~29 mil miniaturas restantes a disco sin tocar ningún Parquet ni recalcular vectores.
4. **Flujos secundarios (en pausa):**
   - Análisis de video por frames.
   - Perfil de usuario basado en los últimos 40 likes.
