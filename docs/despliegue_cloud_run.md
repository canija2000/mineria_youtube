# Publicar la web en Google Cloud Run

Cloud Run corre el contenedor solo mientras hay visitas y lo apaga cuando no las hay. Con poco
tráfico (una demo de curso) puede quedar dentro del nivel gratuito. La imagen trae adentro los
datos y el modelo, así que al despertar no descarga nada.

```
deploy/cloudrun/preparar.sh  ──►  build/cloudrun/ (código + data/ + modelo/, ~2 GB)
deploy/cloudrun/desplegar.sh ──►  Cloud Build construye la imagen ──► Cloud Run la publica en https://mapa-youtube-….run.app
```

## 0. Una sola vez

1. **Cuenta y proyecto.**
   1. Entra a https://console.cloud.google.com y crea un proyecto (por ejemplo `mapa-youtube`). Anota su **ID**, que aparece debajo del nombre.
   2. Activa la **facturación** del proyecto: *Billing → Link a billing account*. Pide tarjeta aunque no se cobre nada dentro del nivel gratuito.
   3. Crea una **alerta de presupuesto**: *Billing → Budgets & alerts → Create budget*, por ejemplo 5 USD. Si algo se sale de lo esperado, te llega un correo.
2. **Instalar y conectar `gcloud`** en el Mac:

   ```bash
   brew install --cask google-cloud-sdk
   ```
   ```bash
   gcloud auth login
   ```
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project <ID-del-proyecto>
   ```

3. **Datos y modelo locales.**
   - Datos: `python3 -m analysis.paquete`, si no tienes `mineria_youtube_datos.zip` en la raíz.
   - Modelo: EmbeddingGemma ya tiene que estar en la caché de Hugging Face, y lo está si alguna vez levantaste el servidor en tu Mac. Si no, `preparar.sh` lo descarga con tu sesión de `hf auth login`.

## 1. Publicar

```bash
deploy/cloudrun/desplegar.sh <ID-del-proyecto>
```

El script:

1. arma `build/cloudrun/` con `preparar.sh`, sin tocar el repo (`build/` está en `.gitignore`);
2. crea, si no existe, el bucket `<ID-del-proyecto>-mapa-persistente` y le da permiso al servicio para escribir en él (ver sección 3);
3. lee `YOUTUBE_API_KEY` del `.env`, sin mostrarla;
4. corre `gcloud run deploy --source`: sube el contexto (~2 GB) y **Cloud Build** construye la imagen en la nube, para linux/amd64 como pide Cloud Run, así que no hay que construirla en el Mac;
5. la publica con 2 CPU, 8 GB de RAM, entre 0 y 1 instancias y acceso público, con el bucket montado en `/mnt/persistente`. El contenedor usa ~5,3 GB porque en CPU el modelo vive en RAM; con 4 GB muere por falta de memoria;
6. fija una limpieza automática de imágenes viejas: quedan solo las 2 últimas (`deploy/cloudrun/limpieza.json`).

La primera vez `gcloud` pregunta si crea el repositorio de imágenes (`cloud-run-source-deploy`): responde que sí. Todo toma ~10–20 minutos, sobre todo por la subida y la instalación de dependencias. Al final imprime la URL, del estilo `https://mapa-youtube-xxxxx-uc.a.run.app`.

Para **actualizar** después de cambiar código, se vuelve a correr el mismo comando.

## 2. Ver que funcione

- **Logs.** Desde la consola, en *Cloud Run → mapa-youtube → Logs*, o desde la terminal:

  ```bash
  gcloud run services logs read mapa-youtube --region us-central1 --project <ID-del-proyecto>
  ```

  Al arrancar deben aparecer `[catálogo] mapa de miniaturas …`, `[catálogo] mapa …: 36.0xx videos` y `Uvicorn running on http://0.0.0.0:8080`. Probada en local con los mismos límites, la imagen arranca en ~30 s, responde un concepto en ~1 s y una búsqueda por imagen en ~5 s.

- **Primera visita.** La primera visita después de un rato sin uso tarda ~1 minuto, porque el contenedor despierta y carga el catálogo y el modelo. Las siguientes son normales.

| síntoma | causa y arreglo |
|---|---|
| `Memory limit of … MiB exceeded` | la RAM no alcanza: el script ya usa `--memory 8Gi`; no la bajes de ~6 GB |
| `Billing account … is not enabled` / `PERMISSION_DENIED` al habilitar APIs | falta activar la facturación del proyecto (paso 0.1) |
| el deploy falla en el *build* | el log de Cloud Build tiene el detalle: *Cloud Build → History* |
| la web carga pero "No se pudo procesar el video: Falta YOUTUBE_API_KEY" | el `.env` no tenía la llave; agrégala y vuelve a desplegar |
| `gcloud crashed (ReadTimeout) … storage.googleapis.com … read timeout=300` | la subida de ~2 GB tardó más que el límite de gcloud; el script ya lo sube a 30 min (`CLOUDSDK_CORE_HTTP_TIMEOUT`). Si vuelve a pasar, reintenta con una conexión más estable |
| `Permission denied` / `403` al escribir en `/mnt/persistente` | la cuenta de servicio no tiene permiso en el bucket; vuelve a correr el script, que lo asigna |

## 3. Persistencia de los videos nuevos

Cada contenedor nace desde la imagen, así que lo que escribe se pierde cuando Cloud Run lo apaga o cuando se vuelve a desplegar. Para que los videos que la gente agrega no se pierdan, el script monta un **bucket de Cloud Storage** como carpeta (`/mnt/persistente`), y el servidor (`server/persistencia.py`):

- **al arrancar**, trae de ahí el catálogo de videos recibidos (`webapp_catalog.sqlite3`). La primera vez lo siembra con el que viene en la imagen;
- **después de cada video nuevo**, deja ahí una copia del catálogo;
- **guarda ahí las miniaturas** de los videos ingresados (`thumbs/`).

El catálogo no se usa directo sobre el bucket, porque esos montajes no manejan bien una base de datos SQLite; se copia completo, y pesa pocos MB. Funciona porque hay una sola instancia (`--max-instances 1`): nunca hay dos contenedores escribiendo a la vez.

Para ver lo guardado, en la consola entra a *Cloud Storage → Buckets → `<ID>-mapa-persistente`*, o desde la terminal:

```bash
gcloud storage ls -l gs://<ID-del-proyecto>-mapa-persistente/
```

En local no cambia nada: sin la variable `MAPA_PERSISTENTE` todo queda en `data/`.

## 4. Costos: qué se cobra y dónde verlo

**Dónde mirar:**

- ***Billing → Reports*:** el gasto del mes. Filtra por el proyecto y agrupa por *Service* para ver cuánto va de Cloud Run, Artifact Registry, Cloud Storage, etc. El uso dentro del nivel gratuito aparece, pero sin cobro.
- ***Billing → Overview*:** el total del mes y, si activaste la prueba gratuita de Google Cloud, los créditos que te quedan.
- ***Billing → Budgets & alerts*:** el presupuesto con alerta por correo (paso 0.1). Es lo más útil para no llevarse sorpresas.
- ***Cloud Run → mapa-youtube → Metrics*:** cuántas instancias hubo y el *Billable container instance time*. Con eso calculas cuánto del nivel gratuito vas usando.

Los reportes de facturación se actualizan con algunas horas de retraso.

**Qué se cobra.** Los precios cambian; revisa la página de precios de Cloud Run. A la fecha de esta guía:

- **Nivel gratuito de Cloud Run** (por mes, en regiones como `us-central1`):
  - 180.000 vCPU-segundos, 360.000 GiB-segundos y 2 millones de pedidos.
  - Con 2 CPU y 8 GB lo que se agota primero es la memoria: **~12 horas de uso activo al mes**. Una demo con visitas esporádicas suele quedar adentro.
  - Solo se cobra mientras se atienden pedidos, y sin visitas la instancia se apaga.
- **Imagen guardada** (Artifact Registry): pesa ~3–4 GB comprimida; después de los primeros 0,5 GB gratis son ~0,10 USD por GB al mes. Con la limpieza que deja 2 imágenes, **~0,60 USD/mes** como máximo.
- **Bucket** (Cloud Storage): unos pocos MB de catálogo y miniaturas, dentro de los 5 GB gratis.
- **Salida de datos a internet:** cada carga del mapa son ~2–3 MB. Hay una cuota gratuita mensual; con una demo no debería notarse.
- **Cloud Build:** cada deploy usa algunos minutos de build, dentro de la cuota gratuita diaria.
- **`--max-instances 1`** acota el gasto, y es lo que hace segura la persistencia: un solo contenedor escribe el catálogo.
- **La cuota de la API de YouTube es tuya.** Como la URL es pública, cualquiera puede gastarla agregando videos o perfiles.

## 5. Bajar el servicio

```bash
gcloud run services delete mapa-youtube --region us-central1 --project <ID-del-proyecto>
```

Las imágenes quedan en *Artifact Registry → cloud-run-source-deploy* y los videos guardados en el bucket `<ID>-mapa-persistente`. Bórralos ahí para no pagar su almacenamiento.
