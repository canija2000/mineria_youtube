#!/usr/bin/env bash
# Arma build/cloudrun/, el contexto de la imagen de Cloud Run:
#   código del servidor + data/ (paquete de datos) + modelo/ (copia de EmbeddingGemma de la caché local)
# Uso: deploy/cloudrun/preparar.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/build/cloudrun"
PY="$ROOT/.venv/bin/python3"
[ -x "$PY" ] || PY=python3
cd "$ROOT"

rm -rf "$BUILD"
mkdir -p "$BUILD"
cp -R recolect server web requirements-server.txt "$BUILD/"
cp deploy/cloudrun/Dockerfile "$BUILD/"
find "$BUILD" -name __pycache__ -type d -prune -exec rm -rf {} +

# Datos: el mismo paquete que se comparte para los notebooks
[ -f mineria_youtube_datos.zip ] || "$PY" -m analysis.paquete
unzip -q mineria_youtube_datos.zip -d "$BUILD"

# Modelo: la versión de la caché local de Hugging Face (si no está, se descarga con tu sesión de `hf auth login`).
# -L copia los archivos reales en vez de los enlaces de la caché.
SNAP=$("$PY" - <<'EOF'
from huggingface_hub import snapshot_download
from recolect.embedding import MODEL_NAME
try:
    print(snapshot_download(MODEL_NAME, local_files_only=True))
except Exception:
    print(snapshot_download(MODEL_NAME))
EOF
)
cp -RL "$SNAP" "$BUILD/modelo"

echo "contexto listo en $BUILD ($(du -sh "$BUILD" | cut -f1))"
