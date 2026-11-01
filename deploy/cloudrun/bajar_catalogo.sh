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

if ! gcloud storage cp "gs://$BUCKET/webapp_catalog.sqlite3" "$TMP/nube.sqlite3" --project "$PROYECTO" >/dev/null 2>&1; then
  echo "[catálogo] el bucket gs://$BUCKET todavía no tiene catálogo; nada que bajar"
  exit 0
fi
cd "$ROOT"
"$PY" -m recolect.received "$TMP/nube.sqlite3" data
