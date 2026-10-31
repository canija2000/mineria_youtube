#!/bin/sh
# Actualiza el código, reconstruye la imagen y (re)levanta el contenedor.
# Se corre en el servidor, desde cualquier carpeta:  sh deploy/oracle/levantar.sh
# Necesita data/ (el paquete de datos descomprimido) y .env en la raíz del repo.
set -e
cd "$(dirname "$0")/../.."

git pull --ff-only
docker build -t mapa-youtube .
docker rm -f mapa 2>/dev/null || true

# Solo escucha en 127.0.0.1: desde internet se entra por Caddy (puertos 80/443).
# data/ queda en el disco de la VM, así que los videos ingresados y el modelo sobreviven.
docker run -d --name mapa --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/data:/app/data" \
  --env-file .env \
  mapa-youtube

echo "Contenedor levantado. Logs (Ctrl+C para salir; el servidor sigue corriendo):"
docker logs -f mapa
