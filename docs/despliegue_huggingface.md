# Publicar la web en Hugging Face Spaces

> **Ojo:** desde fines de 2026 los Spaces con Docker en CPU gratis requieren la suscripción PRO (al crear el Space responde `402 Payment Required`). La alternativa gratis está en [despliegue_oracle.md](despliegue_oracle.md).

Un *Space* con SDK **Docker** construye el `Dockerfile` del repo y deja la web en
`https://<usuario>-<space>.hf.space`. El plan gratuito (CPU basic) alcanza: el servidor usa ~2–3 GB de RAM.

## 0. Lo que hay que tener a mano

- Una cuenta de Hugging Face y un **token de escritura** (huggingface.co → Settings → Access Tokens → *New token*, tipo *Write*).
- La licencia de EmbeddingGemma aceptada con esa cuenta, si el modelo la pide: abre la página del modelo en Hugging Face y acepta los términos.
- El paquete de datos: `python3 -m analysis.paquete` (escribe `mineria_youtube_datos.zip`, ~650 MB).
- La `YOUTUBE_API_KEY` del `.env`.

## 1. Crear el Space y subir el código

```bash
hf auth login                      # pega el token de escritura
python3 deploy/subir_space.py <usuario>/mapa-youtube --datos mineria_youtube_datos.zip
```

El script:

- crea el Space (privado; `--publico` para que cualquiera lo vea);
- sube solo lo que usa el Dockerfile: `recolect/`, `server/`, `web/`, `remap.py`, `requirements-server.txt`, `Dockerfile` y `.dockerignore`;
- sube `deploy/space_README.md` como `README.md` del Space. El encabezado YAML de ese archivo es la configuración del Space: `sdk: docker` y `app_port: 8000`;
- con `--datos`, sube el zip a un Dataset privado `<usuario>/mapa-youtube-datos` e imprime la `DATA_URL` que hay que usar.

También se puede hacer a mano desde huggingface.co → **New Space** → SDK *Docker* → *Blank*, y subir esos mismos archivos con *Files → Add file*.

## 2. Secretos y variables

En la página del Space: **Settings → Variables and secrets**.

| nombre | tipo | valor |
|---|---|---|
| `YOUTUBE_API_KEY` | secret | la del `.env` (para agregar videos y perfiles) |
| `HF_TOKEN` | secret | un token de lectura de tu cuenta: baja el modelo y el Dataset privado |
| `DATA_URL` | secret o variable | la que imprimió el script, `https://huggingface.co/datasets/<usuario>/mapa-youtube-datos/resolve/main/mineria_youtube_datos.zip` |

Cada cambio de secretos reinicia el Space.

## 3. Ver que funcione

- Pestaña **Logs** del Space:
  - *Build*: instalación de dependencias (~5–10 min la primera vez).
  - *Container*: arranque.
- Si todo va bien, en *Container* aparecen estas líneas, en orden:
  1. `[datos] descargando el paquete…` y `[datos] listo`;
  2. la descarga del modelo;
  3. `[catálogo] mapa …: 36.0xx videos`;
  4. `Uvicorn running on http://0.0.0.0:8000`.
- Luego la pestaña **App** muestra la web.

Errores típicos:

| síntoma en los logs | causa |
|---|---|
| `data/ está vacío y no hay DATA_URL` | falta el secreto `DATA_URL` |
| `401` / `403` al descargar | falta `HF_TOKEN`, o el token no tiene acceso al Dataset o al modelo |
| `lo descargado no es un zip` | la URL no es de descarga directa (pasa con enlaces normales de Drive) |
| la app no carga pero uvicorn corre | el puerto de `app_port` no coincide con el de uvicorn (8000) |

## 4. Lo que hay que saber del plan gratuito

- **El disco no es persistente.** En cada reinicio o redeploy se vuelven a descargar los datos (~650 MB) y el modelo, así que el arranque toma unos minutos. Los videos que la gente agrega por la web (SQLite y miniaturas) se pierden.
- **Almacenamiento persistente (de pago).** Se activa en *Settings*, queda montado en `/data` y se usa definiendo las variables `MAPA_DATA=/data` y `HF_HOME=/data/hf`. Así los datos se descargan una sola vez y los videos ingresados se conservan.
- **El Space se duerme** tras un tiempo sin visitas. La primera visita después lo despierta, lo que toma unos minutos.
- **La cuota de la API de YouTube es tuya.** Si el Space es público, cualquiera puede gastarla agregando videos o perfiles.

## 5. Actualizar

Después de cambiar el código, basta con volver a correr `python3 deploy/subir_space.py <usuario>/mapa-youtube`, sin `--datos` si los datos no cambiaron. El Space se reconstruye solo.
