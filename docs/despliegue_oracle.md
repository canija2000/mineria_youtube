# Publicar la web en Oracle Cloud (Always Free)

Oracle regala una máquina virtual ARM (Ampere A1) de hasta 4 núcleos y 24 GB de RAM, con disco
persistente. Ahí corremos el mismo contenedor Docker del repo, detrás de **Caddy**, que entrega
la web por los puertos 80/443 y saca el certificado HTTPS solo.

```
internet ──► :80/:443 Caddy ──► 127.0.0.1:8000 contenedor "mapa" ──► ~/mineria_youtube/data (disco de la VM)
```

> Esta guía no se ha probado de punta a punta en Oracle. Los pasos donde la gente suele trabarse
> van marcados con ⚠️.

---

## 1. Crear la cuenta

1. Entra a https://cloud.oracle.com y elige **Sign up**.
2. Pide tarjeta de crédito solo para verificar identidad. No cobra mientras uses recursos *Always Free*.
3. ⚠️ **La región de origen (*home region*) no se puede cambiar después.** Elige una cercana. A veces una región no tiene máquinas ARM disponibles; si te pasa en el paso 3, la solución está ahí.

## 2. Llave SSH (en tu Mac)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/oracle_mapa -C "mapa-youtube"
```

Se crean dos archivos: `~/.ssh/oracle_mapa`, que es privado y no se comparte, y `~/.ssh/oracle_mapa.pub`, que se sube a Oracle.

## 3. Crear la red y la máquina virtual

**Primero la red.** ⚠️ Si la red se crea desde el formulario de la instancia, la consola no deja activar la IP pública. Por eso se crea antes, con su asistente:

1. Ve a **☰ → Networking → Virtual cloud networks → Actions → Start VCN Wizard → Create VCN with Internet Connectivity**.
2. Ponle de nombre `vcn-mapa`, deja el resto por defecto y elige **Next → Create**. Se crean la subred pública, el *internet gateway* y las rutas.

**Después la máquina:** **☰ → Compute → Instances → Create instance**.

| campo | valor |
|---|---|
| Name | `mapa-youtube` |
| Create in compartment | el que viene (`<tu-usuario> (root)`) |
| Placement → Availability domain | **AD 1**. Si al crear dice *Out of capacity*, vuelve y prueba AD 2 o AD 3 |
| Advanced options → Capacity type | **On-demand capacity** (el que viene). *Preemptible* no: Oracle puede apagarla cuando quiera |
| Shape (**primero**) | *Change shape* → **Ampere** → `VM.Standard.A1.Flex`, **2 OCPU / 12 GB** (alcanza de sobra; el máximo gratis es 4/24) |
| Image (**después**) | *Change image* → **Canonical Ubuntu 24.04**, la normal, no la *Minimal*. Con la shape Ampere ya elegida, Oracle usa sola la versión aarch64. Si eliges la imagen antes que la shape, aparece un aviso de imagen incompatible |
| Security | todo apagado (*Shielded instance* y *Confidential computing*). El aviso de que no se puede activar *confidential computing* es normal con Ampere |
| Networking | **Select existing virtual cloud network** → `vcn-mapa`; subnet: la **pública**; **Automatically assign public IPv4 address** activado |
| Add SSH keys | *Upload public key file* → `~/.ssh/oracle_mapa.pub`. Otra opción: *Generate a key pair for me* y **Download private key** antes de crear la instancia (solo se muestra una vez); guárdala como `~/.ssh/oracle_mapa` con `chmod 600` |
| Boot volume | *Specify a custom boot volume size* → **100 GB** (el gratis llega a 200 GB en total) |

Pulsa **Create** y anota la **Public IP address** que aparece cuando el estado pasa a *Running*.

⚠️ **"Out of capacity for shape VM.Standard.A1.Flex"** es muy común. Tienes tres salidas:

- reintentar más tarde;
- probar otro *Availability Domain* en el formulario;
- pasar la cuenta a **Pay As You Go** (*Billing → Upgrade*). Sigue sin cobrar dentro de los límites gratis, la capacidad ARM suele aparecer y Oracle no recupera la máquina por inactividad. Si lo haces, configura una alerta de presupuesto en *Billing → Budgets*.

## 4. Abrir los puertos 80 y 443

Hay **dos** firewalls y hay que abrir ambos.

**a) El de Oracle (Security List):** en la instancia, abre la *subnet* y entra a **Security Lists → Default Security List → Add Ingress Rules**. Agrega dos reglas:

| Source CIDR | IP Protocol | Destination Port |
|---|---|---|
| `0.0.0.0/0` | TCP | `80` |
| `0.0.0.0/0` | TCP | `443` |

El puerto 22 (SSH) ya viene abierto.

**b) El de Ubuntu (iptables):** ⚠️ es el que más se olvida. Las imágenes de Oracle traen reglas que bloquean todo menos SSH. Se configura en el paso 5, ya conectado.

## 5. Conectarse y preparar el servidor

```bash
ssh -i ~/.ssh/oracle_mapa ubuntu@<IP-pública>
```

Ya dentro del servidor:

```bash
# actualizar e instalar herramientas
sudo apt update && sudo apt -y upgrade
sudo apt -y install git unzip

# firewall de Ubuntu: permitir 80 y 443 (y que sobreviva a reinicios)
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit          # salir y volver a entrar para que el grupo docker aplique
```

```bash
ssh -i ~/.ssh/oracle_mapa ubuntu@<IP-pública>
docker run --rm hello-world      # debe imprimir "Hello from Docker!"
```

## 6. Código, datos y secretos

En el servidor:

```bash
git clone https://github.com/canija2000/mineria_youtube.git
cd mineria_youtube
git checkout codex/pipeline-webapp   # hasta que el PR se una a main
```

Si el repo es privado, `git clone` pide usuario y contraseña. La contraseña es un *Personal Access Token* de GitHub (GitHub → Settings → Developer settings → Tokens), no tu contraseña normal.

**Datos.** Desde tu Mac, en otra terminal, copia el paquete (`python3 -m analysis.paquete` si no lo tienes):

```bash
scp -i ~/.ssh/oracle_mapa mineria_youtube_datos.zip ubuntu@<IP-pública>:~/mineria_youtube/
```

De vuelta en el servidor:

```bash
cd ~/mineria_youtube
unzip -o mineria_youtube_datos.zip && rm mineria_youtube_datos.zip
ls data          # embeddings_consolidated.parquet, maps/, maps_image/, ...
```

**Secretos.** En el servidor:

```bash
cat > .env <<'EOF'
YOUTUBE_API_KEY=pega_aquí_tu_llave
HF_TOKEN=pega_aquí_un_token_de_lectura_de_hugging_face
EOF
chmod 600 .env
```

`HF_TOKEN` hace falta solo si EmbeddingGemma pide aceptar la licencia de Gemma. En ese caso, acéptala antes con tu cuenta en la página del modelo.

## 7. Construir y levantar el contenedor

```bash
sh deploy/oracle/levantar.sh
```

El script hace todo en orden:

1. `git pull`;
2. construye la imagen, que en ARM la primera vez toma unos 10–15 min;
3. levanta el contenedor `mapa` con reinicio automático, escuchando solo en `127.0.0.1:8000` y con `data/` montado.

Al final muestra los logs. La primera vez descarga el modelo (queda guardado en `data/hf`). Está listo cuando aparecen estas líneas:

```
[catálogo] mapa ...: 36.0xx videos
Uvicorn running on http://0.0.0.0:8000
```

Sal de los logs con `Ctrl+C`; el contenedor sigue corriendo. Para comprobar que responde:

```bash
curl -s localhost:8000/health
```

## 8. Caddy: la web desde internet

```bash
sudo apt -y install caddy
sudo cp deploy/oracle/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Abre **`http://<IP-pública>`** en el navegador y debería aparecer el mapa.

**HTTPS con un dominio gratis (opcional, recomendado):**

1. En https://www.duckdns.org entra con tu cuenta de GitHub o Google, crea un subdominio (p. ej. `mapa-youtube`) y ponle la IP pública de la VM.
2. En `/etc/caddy/Caddyfile`, reemplaza el bloque `:80 { … }` por el de dominio que está comentado en el archivo, con tu subdominio.
3. Ejecuta `sudo systemctl reload caddy`. Caddy pide el certificado a Let's Encrypt solo, y en un minuto funciona **`https://mapa-youtube.duckdns.org`**.

## 9. Mantenimiento

| tarea | comando (en el servidor, dentro de `~/mineria_youtube`) |
|---|---|
| actualizar a la última versión del código | `sh deploy/oracle/levantar.sh` |
| ver logs | `docker logs -f mapa` |
| reiniciar | `docker restart mapa` |
| estado / memoria | `docker ps` · `docker stats mapa` |
| recalcular el mapa con los videos ingresados | `docker exec mapa python remap.py --min-new 50` (el servidor toma la versión nueva solo) |
| respaldar lo ingresado por la web | desde tu Mac: `scp -i ~/.ssh/oracle_mapa ubuntu@<IP>:~/mineria_youtube/data/webapp_catalog.sqlite3 .` |

Para recalcular el mapa todas las noches, en el servidor ejecuta `crontab -e` y agrega:

```
0 4 * * * docker exec mapa python remap.py --min-new 50 >> /home/ubuntu/remap.log 2>&1
```

## 10. Si algo falla

| síntoma | revisar |
|---|---|
| `ssh` se queda pegado | IP correcta, llave correcta (`-i`), regla del puerto 22 en la Security List |
| `http://<IP>` no carga | las **dos** capas del paso 4 (Security List **y** iptables); `sudo systemctl status caddy` |
| Caddy responde 502 | el contenedor no está arriba: `docker ps`, `docker logs mapa` |
| `docker build` falla instalando paquetes | copia el error; puede faltar alguna rueda para ARM de una versión fija de `requirements-server.txt` |
| el contenedor se reinicia en bucle | `docker logs mapa`: típicamente falta `data/` o el modelo pide `HF_TOKEN` |
| HTTPS no funciona | el dominio apunta a la IP (`ping mapa-youtube.duckdns.org`), puerto 443 abierto, `journalctl -u caddy` |

## Seguridad básica

- Solo quedan abiertos los puertos 22, 80 y 443. El servidor de Python no se expone directo: escucha en `127.0.0.1`.
- `.env` tiene la llave de YouTube. No lo subas a git (ya está en `.gitignore`) y déjalo con `chmod 600`.
- Si la web es pública, cualquiera puede gastar tu cuota de YouTube agregando videos o perfiles.
- Ubuntu instala solo las actualizaciones de seguridad (`unattended-upgrades` viene activo).
