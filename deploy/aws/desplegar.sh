#!/usr/bin/env bash
# Publica la web en una máquina EC2 de AWS (ver docs/despliegue_aws.md).
# Se corre en el Mac. Usa la misma imagen que Cloud Run (datos y modelo adentro), pero la
# construye en la máquina, que es x86 como Cloud Run (el Mac es ARM).
#
# Uso: deploy/aws/desplegar.sh <IP-pública> [dominio]
#   dominio (opcional): p. ej. mapa-youtube.duckdns.org apuntando a la IP; Caddy saca el HTTPS solo.
#   La llave SSH es ~/.ssh/aws_mapa.pem; para usar otra: MAPA_SSH_KEY=/ruta/llave.pem deploy/aws/desplegar.sh ...
set -euo pipefail

IP="${1:?falta la IP pública de la máquina EC2}"
DOMINIO="${2:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KEY="${MAPA_SSH_KEY:-$HOME/.ssh/aws_mapa.pem}"
[ -f "$KEY" ] || { echo "no encuentro la llave SSH $KEY (ver docs/despliegue_aws.md, paso 3)"; exit 1; }
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "ubuntu@$IP")
PY="$ROOT/.venv/bin/python3"
[ -x "$PY" ] || PY=python3

# 1. Traer los videos que llegaron por la web: la imagen nueva lleva el catálogo local,
#    y sin esto los que solo están en la máquina no entrarían a un mapa recalculado.
"$ROOT/deploy/aws/bajar_catalogo.sh" "$IP"
(cd "$ROOT" && "$PY" - <<'EOF'
import json, pandas as pd
from recolect.received import ReceivedStore, store_path
version = json.load(open("data/maps/current.json"))["version"]
en_mapa = set(pd.read_parquet(f"data/maps/{version}/coords.parquet", columns=["youtube_id"])["youtube_id"])
fuera = sum(1 for r in ReceivedStore(store_path("data")).all() if r["youtube_id"] not in en_mapa)
if fuera:
    print(f"[mapa] {fuera} videos recibidos no están en el mapa {version}: se ubican encima, sin moverlo. "
          f"Para incluirlos: .venv/bin/python remap.py y volver a desplegar.")
EOF
)

# 2. Contexto de la imagen (código + data/ + modelo/), el mismo de Cloud Run
"$ROOT/deploy/cloudrun/preparar.sh"

# 3. Subirlo. rsync solo manda lo que cambió: la primera vez son ~2 GB, después pocos MB.
echo "[subida] build/cloudrun -> $IP:~/mapa/contexto"
"${SSH[@]}" "mkdir -p ~/mapa/contexto ~/mapa/persistente"
rsync -a --delete -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  "$ROOT/build/cloudrun/" "ubuntu@$IP:mapa/contexto/"

# La llave de YouTube va en un archivo con permisos 600 en la máquina (no se muestra).
# Sin ella la web funciona, pero no se pueden agregar videos ni perfiles nuevos.
YT_KEY="$(grep -E '^YOUTUBE_API_KEY=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"
printf 'YOUTUBE_API_KEY=%s\n' "$YT_KEY" | "${SSH[@]}" "umask 077 && cat > ~/mapa/.env"

# 4. En la máquina: instalar lo que falte, construir y levantar
"${SSH[@]}" "DOMINIO='$DOMINIO' bash -s" <<'REMOTO'
set -euo pipefail

# Swap de 4 GB: el contenedor usa ~5,3 de los 8 GB, y construir la imagen nueva mientras
# corre la vieja puede pasarse de memoria.
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
# BBR en vez de cubic: entre Chile y us-east-2 hay ~175 ms de ida y vuelta y algo de pérdida de
# paquetes, y con cubic el mapa (2,6 MB) tardaba 30–45 s en bajar; con BBR, ~4 s.
if [ "$(sysctl -n net.ipv4.tcp_congestion_control)" != bbr ]; then
  printf 'net.core.default_qdisc=fq\nnet.ipv4.tcp_congestion_control=bbr\n' | sudo tee /etc/sysctl.d/90-bbr.conf >/dev/null
  sudo sysctl -q --system
fi
if ! command -v docker >/dev/null; then
  echo "[servidor] instalando Docker"
  curl -fsSL https://get.docker.com | sudo sh >/dev/null
fi
if ! command -v caddy >/dev/null; then
  echo "[servidor] instalando Caddy"
  sudo apt-get update -qq && sudo apt-get install -y -qq caddy >/dev/null
fi

echo "[servidor] construyendo la imagen (la primera vez ~10 min)"
sudo docker build -q -t mapa-youtube ~/mapa/contexto >/dev/null
sudo docker rm -f mapa >/dev/null 2>&1 || true
# Solo escucha en 127.0.0.1: desde internet se entra por Caddy (80/443).
# ~/mapa/persistente queda en el disco de la máquina: ahí se guardan el catálogo de videos
# recibidos y sus miniaturas, y sobreviven a cada despliegue (server/persistencia.py).
sudo docker run -d --name mapa --restart unless-stopped \
  -p 127.0.0.1:8000:8000 -e PORT=8000 \
  -e MAPA_PERSISTENTE=/mnt/persistente -v "$HOME/mapa/persistente:/mnt/persistente" \
  --env-file "$HOME/mapa/.env" \
  mapa-youtube >/dev/null
sudo docker image prune -f >/dev/null   # borra las imágenes viejas (3–4 GB cada una)

# Sin dominio se mantiene el Caddyfile que ya estaba (así un despliegue sin dominio no apaga el HTTPS)
if [ -n "$DOMINIO" ] || ! grep -q reverse_proxy /etc/caddy/Caddyfile; then
  if [ -n "$DOMINIO" ]; then SITIO="$DOMINIO"; else SITIO=":80"; fi
  printf '%s {\n\tencode gzip\n\treverse_proxy 127.0.0.1:8000\n}\n' "$SITIO" | sudo tee /etc/caddy/Caddyfile >/dev/null
  sudo systemctl reload caddy || sudo systemctl restart caddy
fi

echo -n "[servidor] esperando que cargue el mapa"
for _ in $(seq 60); do
  if curl -fs localhost:8000/health >/dev/null; then echo " listo"; exit 0; fi
  echo -n "."; sleep 5
done
echo; echo "no respondió en 5 minutos; últimos logs:"; sudo docker logs --tail 40 mapa; exit 1
REMOTO

if [ -n "$DOMINIO" ]; then echo "publicado en https://$DOMINIO"; else echo "publicado en http://$IP"; fi
