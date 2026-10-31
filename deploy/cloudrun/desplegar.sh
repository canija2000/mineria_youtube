#!/usr/bin/env bash
# Publica la web en Google Cloud Run (ver docs/despliegue_cloud_run.md).
# Uso: deploy/cloudrun/desplegar.sh <id-del-proyecto> [región]
set -euo pipefail

PROYECTO="${1:?falta el id del proyecto de Google Cloud}"
REGION="${2:-us-central1}"   # región del nivel gratuito de Cloud Run
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

"$ROOT/deploy/cloudrun/preparar.sh"

# La llave de YouTube se lee del .env (sin mostrarla); sin ella la web funciona,
# pero no se pueden agregar videos ni perfiles nuevos.
KEY="$(grep -E '^YOUTUBE_API_KEY=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"

# --source: Cloud Build construye la imagen en la nube (linux/amd64), así no hay que construirla en el Mac.
# --max-instances 1: cada instancia tiene su propio SQLite y así el gasto queda acotado.
# --min-instances 0: sin visitas se apaga y no cobra; la primera visita después tarda ~1 min.
# --memory 8Gi: en CPU el modelo vive en RAM; el contenedor usa ~5,3 GB (con 4 GB muere por falta de memoria).
gcloud run deploy mapa-youtube \
  --source "$ROOT/build/cloudrun" \
  --project "$PROYECTO" --region "$REGION" \
  --memory 8Gi --cpu 2 --cpu-boost \
  --min-instances 0 --max-instances 1 --concurrency 20 --timeout 300 \
  --allow-unauthenticated \
  ${KEY:+--set-env-vars "YOUTUBE_API_KEY=$KEY"}
