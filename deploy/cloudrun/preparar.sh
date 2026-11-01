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

# Datos: el mismo paquete que se comparte para los notebooks. Se rehace si no existe o si trae
# otra versión de mapa que la vigente en data/ (si no, después de `remap.py` se subiría el mapa viejo).
if ! "$PY" - <<'EOF'
import json, sys, zipfile
from pathlib import Path
try:
    z = zipfile.ZipFile("mineria_youtube_datos.zip")
except FileNotFoundError:
    sys.exit(1)
for carpeta in ("maps", "maps_image"):
    vigente = Path(f"data/{carpeta}/current.json")
    if vigente.exists() and json.loads(z.read(f"data/{carpeta}/current.json")) != json.loads(vigente.read_text()):
        print(f"[paquete] el zip trae otra versión de data/{carpeta}; se rehace")
        sys.exit(1)
EOF
then
  "$PY" -m analysis.paquete
fi
unzip -q mineria_youtube_datos.zip -d "$BUILD"
# El catálogo de recibidos cambia seguido (y desplegar.sh le suma el del bucket): va siempre el actual
cp data/webapp_catalog.sqlite3 "$BUILD/data/"

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
