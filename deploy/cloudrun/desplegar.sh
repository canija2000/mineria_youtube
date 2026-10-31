#!/usr/bin/env bash
# Publica la web en Google Cloud Run (ver docs/despliegue_cloud_run.md).
# Uso: deploy/cloudrun/desplegar.sh <id-del-proyecto> [región]
set -euo pipefail

PROYECTO="${1:?falta el id del proyecto de Google Cloud}"
REGION="${2:-us-central1}"   # región del nivel gratuito de Cloud Run
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUCKET="${PROYECTO}-mapa-persistente"

"$ROOT/deploy/cloudrun/preparar.sh"

# Bucket donde persisten los videos que llegan por la web (catálogo SQLite y miniaturas).
# Sin él, se perderían cada vez que Cloud Run apaga el contenedor o se vuelve a desplegar.
if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROYECTO" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --project "$PROYECTO" --location "$REGION" \
    --uniform-bucket-level-access
fi
# El servicio corre con la cuenta de servicio por defecto de Compute; necesita leer y escribir en el bucket.
NUMERO="$(gcloud projects describe "$PROYECTO" --format='value(projectNumber)')"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --project "$PROYECTO" \
  --member "serviceAccount:${NUMERO}-compute@developer.gserviceaccount.com" \
  --role roles/storage.objectUser >/dev/null

# La llave de YouTube se lee del .env (sin mostrarla); sin ella la web funciona,
# pero no se pueden agregar videos ni perfiles nuevos.
KEY="$(grep -E '^YOUTUBE_API_KEY=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"
ENV_VARS="MAPA_PERSISTENTE=/mnt/persistente${KEY:+,YOUTUBE_API_KEY=$KEY}"

# El volumen se agrega solo si el servicio todavía no lo tiene (agregarlo dos veces da error).
VOLUMEN=()
if ! gcloud run services describe mapa-youtube --project "$PROYECTO" --region "$REGION" \
     --format='value(spec.template.spec.volumes[].name)' 2>/dev/null | grep -q persistente; then
  VOLUMEN=(--add-volume "name=persistente,type=cloud-storage,bucket=$BUCKET"
           --add-volume-mount "volume=persistente,mount-path=/mnt/persistente")
fi

# --source: Cloud Build construye la imagen en la nube (linux/amd64), así no hay que construirla en el Mac.
# --max-instances 1: un solo contenedor escribe el catálogo y el gasto queda acotado.
# --min-instances 0: sin visitas se apaga y no cobra; la primera visita después tarda ~1 min.
# --memory 8Gi: en CPU el modelo vive en RAM; el contenedor usa ~5,3 GB (con 4 GB muere por falta de memoria).
# --add-volume*: monta el bucket en /mnt/persistente (requiere el entorno de ejecución gen2).
gcloud run deploy mapa-youtube \
  --source "$ROOT/build/cloudrun" \
  --project "$PROYECTO" --region "$REGION" \
  --memory 8Gi --cpu 2 --cpu-boost \
  --min-instances 0 --max-instances 1 --concurrency 20 --timeout 300 \
  --execution-environment gen2 \
  ${VOLUMEN[@]+"${VOLUMEN[@]}"} \
  --allow-unauthenticated \
  --set-env-vars "$ENV_VARS"

# Cada despliegue guarda una imagen nueva de 3–4 GB en Artifact Registry y se cobra su almacenamiento.
# Esta política (deploy/cloudrun/limpieza.json) deja solo las 2 últimas; Google la aplica una vez al día.
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
  --project "$PROYECTO" --location "$REGION" \
  --policy "$ROOT/deploy/cloudrun/limpieza.json" --no-dry-run >/dev/null \
  || echo "aviso: no se pudo fijar la limpieza de imágenes viejas (ver docs/despliegue_cloud_run.md)"
