#!/usr/bin/env bash
# Suma al catálogo local (data/webapp_catalog.sqlite3) los videos que llegaron por la web en la
# máquina EC2. Correrlo antes de `.venv/bin/python remap.py`, para que el mapa nuevo los incluya;
# desplegar.sh también lo corre. Solo agrega: no borra ni pisa nada del catálogo local.
# Uso: deploy/aws/bajar_catalogo.sh <IP-pública>   (llave: ~/.ssh/aws_mapa.pem o MAPA_SSH_KEY)
set -euo pipefail

IP="${1:?falta la IP pública de la máquina EC2}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KEY="${MAPA_SSH_KEY:-$HOME/.ssh/aws_mapa.pem}"
PY="$ROOT/.venv/bin/python3"
[ -x "$PY" ] || PY=python3

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if ! scp -q -i "$KEY" -o StrictHostKeyChecking=accept-new \
     "ubuntu@$IP:mapa/persistente/webapp_catalog.sqlite3" "$TMP/nube.sqlite3" 2>/dev/null; then
  echo "[catálogo] la máquina todavía no tiene catálogo; nada que bajar"
  exit 0
fi
cd "$ROOT"
"$PY" -m recolect.received "$TMP/nube.sqlite3" data
