#!/usr/bin/env bash
# Suma al catálogo local (data/webapp_catalog.sqlite3) los videos que llegaron por la web en Cloud Run.
# Correrlo antes de `python3 remap.py`, para que el mapa nuevo los incluya; desplegar.sh también lo corre.
# Solo agrega: no borra ni pisa nada del catálogo local.
# Uso: deploy/cloudrun/bajar_catalogo.sh <id-del-proyecto>
set -euo pipefail

PROYECTO="${1:?falta el id del proyecto de Google Cloud}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUCKET="${PROYECTO}-mapa-persistente"
PY="$ROOT/.venv/bin/python3"
[ -x "$PY" ] || PY=python3

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

OBJETO="gs://$BUCKET/webapp_catalog.sqlite3"
# Solo "no existe" es motivo para seguir sin bajar nada; cualquier otro error (sesión vencida,
# permisos, red) se muestra y detiene el script, para no desplegar o borrar creyendo que no había nada.
if ! LISTA="$(gcloud storage ls "$OBJETO" --project "$PROYECTO" 2>&1)"; then
  if grep -qiE "matched no objects|not found|does not exist" <<<"$LISTA"; then
    echo "[catálogo] el bucket gs://$BUCKET todavía no tiene catálogo; nada que bajar"
    exit 0
  fi
  echo "[catálogo] no se pudo revisar $OBJETO:"; echo "$LISTA"; exit 1
fi
# Sin la verificación de gcloud: en algunos Mac no logra calcular el CRC32C local (da "AAAAAA==") y
# rechaza una descarga buena. La integridad se revisa abajo, con el MD5 que informa Cloud Storage.
CLOUDSDK_STORAGE_CHECK_HASHES=never gcloud storage cp "$OBJETO" "$TMP/nube.sqlite3" --project "$PROYECTO"
MD5_NUBE="$(gcloud storage objects describe "$OBJETO" --project "$PROYECTO" --format='value(md5_hash)')"
MD5_LOCAL="$("$PY" -c 'import base64,hashlib,sys; print(base64.b64encode(hashlib.md5(open(sys.argv[1],"rb").read()).digest()).decode())' "$TMP/nube.sqlite3")"
if [ "$MD5_NUBE" != "$MD5_LOCAL" ]; then
  echo "[catálogo] la copia bajada no coincide con la del bucket (MD5 $MD5_LOCAL vs $MD5_NUBE); no se suma nada"
  exit 1
fi
cd "$ROOT"
"$PY" -m recolect.received "$TMP/nube.sqlite3" data
