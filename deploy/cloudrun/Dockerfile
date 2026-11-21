# Imagen para Google Cloud Run: datos y modelo van adentro, así el contenedor no descarga nada
# cuando despierta (Cloud Run lo apaga sin visitas). El contexto lo arma deploy/cloudrun/preparar.sh:
# código del servidor + data/ (paquete de analysis/paquete.py) + modelo/ (copia local de EmbeddingGemma).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAPA_DATA=/app/data \
    MAPA_MODELO=/app/modelo \
    HF_HUB_OFFLINE=1
WORKDIR /app

# torch solo-CPU: la versión con CUDA pesa ~2 GB más y Cloud Run no tiene GPU en este plan
COPY requirements-server.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1 \
 && pip install -r requirements-server.txt

# Lo pesado y que casi no cambia va primero: un cambio de código no vuelve a copiar 2 GB
COPY modelo/ modelo/
COPY data/ data/
COPY recolect/ recolect/
COPY server/ server/
COPY web/ web/

# Cloud Run define PORT (8080)
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
