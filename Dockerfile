# Servidor + web del Mapa de YouTube.
# La imagen no trae datos ni modelo: los datos se montan en /app/data (volumen) o se descargan
# desde DATA_URL al arrancar, y EmbeddingGemma se baja de Hugging Face la primera vez
# (queda en /app/data/hf, así que con un volumen persistente se descarga una sola vez).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAPA_DATA=/app/data \
    HF_HOME=/app/data/hf
WORKDIR /app

# torch solo-CPU: la versión con CUDA pesa ~2 GB más y el servidor no usa GPU
COPY requirements-server.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1 \
 && pip install -r requirements-server.txt

COPY recolect/ recolect/
COPY server/ server/
COPY web/ web/
COPY remap.py .

# Hugging Face Spaces corre el contenedor como uid 1000 (no root): la carpeta de datos
# tiene que existir y ser escribible para que se puedan descargar datos y modelo.
RUN mkdir -p /app/data && chmod -R 777 /app/data

EXPOSE 8000
# Railway (y otros) definen PORT; en local queda en 8000
CMD ["sh", "-c", "python -m server.datos && exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
