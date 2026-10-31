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
2. lee `YOUTUBE_API_KEY` del `.env`, sin mostrarla;
3. corre `gcloud run deploy --source`: sube el contexto (~2 GB) y **Cloud Build** construye la imagen en la nube, para linux/amd64 como pide Cloud Run, así que no hay que construirla en el Mac;
4. la publica con 2 CPU, 8 GB de RAM, entre 0 y 1 instancias y acceso público. El contenedor usa ~5,3 GB porque en CPU el modelo vive en RAM; con 4 GB muere por falta de memoria.

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

## 3. Costos y límites

Los precios cambian; revisa la página de precios de Cloud Run. A la fecha de esta guía:

- **Nivel gratuito de Cloud Run** (por mes, en regiones como `us-central1`):
  - 180.000 vCPU-segundos, 360.000 GiB-segundos y 2 millones de pedidos.
  - Con 2 CPU y 8 GB lo que se agota primero es la memoria: **~12 horas de uso activo al mes**. Una demo con visitas esporádicas suele quedar adentro.
  - Solo se cobra mientras se atienden pedidos, y sin visitas la instancia se apaga.
- **Imagen guardada** (Artifact Registry): pesa ~3–4 GB comprimida; después de los primeros 0,5 GB gratis son ~0,10 USD por GB al mes, así que **~0,30 USD/mes**.
- **Cloud Build:** cada deploy usa algunos minutos de build, dentro de la cuota gratuita diaria.
- **No hay disco persistente.** Los videos que la gente agrega por la web y sus miniaturas viven solo mientras la instancia está despierta; al apagarse se pierden. El mapa base, que va dentro de la imagen, siempre está.
- **`--max-instances 1`** acota el gasto. Además cada instancia tendría su propio catálogo, así que con más de una los usuarios verían cosas distintas.
- **La cuota de la API de YouTube es tuya.** Como la URL es pública, cualquiera puede gastarla agregando videos o perfiles.

## 4. Bajar el servicio

```bash
gcloud run services delete mapa-youtube --region us-central1 --project <ID-del-proyecto>
```

Las imágenes quedan en *Artifact Registry → cloud-run-source-deploy*; bórralas ahí para no pagar su almacenamiento.
