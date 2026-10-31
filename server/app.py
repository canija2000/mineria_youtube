"""API HTTP para agregar un video y devolver su posición y vecinos."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from recolect.embedding import TextEmbedder, build_text
from recolect.youtube import fetch_profile_videos, fetch_video
from .catalog import VideoCatalog

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
URL_RE = re.compile(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/))([A-Za-z0-9_-]{11})")


class AddVideoRequest(BaseModel):
    video: str = Field(min_length=11, max_length=2048, description="ID o URL de YouTube")
    representation: str = Field(default="text", description="Espacio de representación: text o image")


class ConceptRequest(BaseModel):
    concept: str = Field(min_length=1, max_length=200, description="Palabra o concepto a ubicar en el mapa 3D")


class ImageSearchRequest(BaseModel):
    image_url: str | None = Field(default=None, max_length=4096, description="URL de imagen externa")
    image_data: str | None = Field(default=None, description="Data URL base64 de imagen arrastrada/subida")
    label: str | None = Field(default=None, max_length=200, description="Nombre o etiqueta de la imagen")


class ProfileRequest(BaseModel):
    profile: str = Field(min_length=2, max_length=2048, description="@handle, URL de canal o playlist de YouTube")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MAPA_DATA=data/dev permite probar el servidor con los datos del modo prueba.
    app.state.catalog = VideoCatalog(os.getenv("MAPA_DATA", "data"))
    app.state.embedder = TextEmbedder()
    yield


app = FastAPI(title="YouTube Video Map API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)
# El mapa completo son varios MB de JSON; comprimido pesa bastante menos.
app.add_middleware(GZipMiddleware, minimum_size=1000)


def parse_video_id(value: str) -> str:
    value = value.strip()
    if VIDEO_ID_RE.fullmatch(value):
        return value
    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = parsed.netloc.lower().removeprefix("www.")
    candidate = None
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        candidate = parse_qs(parsed.query).get("v", [None])[0]
        if not candidate:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in {"embed", "shorts"}:
                candidate = parts[1]
    if candidate and VIDEO_ID_RE.fullmatch(candidate):
        return candidate
    match = URL_RE.search(value)
    if match:
        return match.group(1)
    raise HTTPException(status_code=422, detail="Ingresa un ID o una URL válida de YouTube")


@app.get("/health")
def health():
    catalog = app.state.catalog
    catalog.refresh_map()
    return {"status": "ok", **catalog.status()}


@app.get("/map")
def get_map(rep: str = "text"):
    """Todos los puntos del mapa vigente (texto o imagen) para que la web los dibuje."""
    catalog = app.state.catalog
    catalog.refresh_map()
    return catalog.map_payload(representation=rep)


def _save_thumbnail(video_id: str) -> Path:
    """Guarda la miniatura del video en data/thumbs/ (las de los videos ingresados sí se conservan;
    lo que no se guarda son las imágenes sueltas de /images)."""
    import requests
    thumb_dir = Path(os.getenv("MAPA_DATA", "data")) / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / f"{video_id}.jpg"
    if not thumb_path.exists() or thumb_path.stat().st_size <= 1000:
        resp = requests.get(f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg", timeout=15)
        resp.raise_for_status()
        thumb_path.write_bytes(resp.content)
    return thumb_path


def _register_video(video_id: str, metadata: dict, catalog: VideoCatalog) -> None:
    """Agrega al catálogo un video que llega por primera vez: metadatos, vector de texto y miniatura."""
    if catalog.has(video_id):
        return
    vector = app.state.embedder.encode([build_text(metadata)])[0]
    catalog.add(video_id, metadata, vector)
    try:
        _save_thumbnail(video_id)
    except Exception as exc:  # sin miniatura el video igual queda en el catálogo
        print(f"[miniatura] no se pudo guardar la de {video_id}: {exc}", flush=True)


def _encode_video_thumbnail(video_id: str, embedder: TextEmbedder):
    thumb_path = _save_thumbnail(video_id)
    embedder.enable_images()
    return embedder.encode([{"image": [str(thumb_path)]}], batch_size=1, prompt_name=None)[0]


@app.post("/videos", status_code=201)
def add_video(payload: AddVideoRequest):
    video_id = parse_video_id(payload.video)
    rep = (payload.representation or "text").strip().lower()
    catalog = app.state.catalog
    catalog.refresh_map()

    if rep == "image":
        if video_id in catalog.image_positions:
            return catalog.describe_image(video_id)
        try:
            metadata = catalog.metadata.get(video_id) or fetch_video(video_id)
            if metadata is None:
                raise HTTPException(status_code=404, detail="El video no existe o no está disponible")
            _register_video(video_id, metadata, catalog)   # queda guardado igual que en modo texto
            img_vec = _encode_video_thumbnail(video_id, app.state.embedder)
            return catalog.describe_image(video_id, metadata=metadata, image_vector=img_vec)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"No se pudo procesar la miniatura: {exc}") from exc

    if catalog.has(video_id):
        return catalog.describe(video_id)
    try:
        metadata = fetch_video(video_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="El video no existe o no está disponible")
        _register_video(video_id, metadata, catalog)
        return catalog.describe(video_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo procesar el video: {exc}") from exc


@app.post("/concepts")
def locate_concept(payload: ConceptRequest):
    query = payload.concept.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Ingresa una palabra o concepto")
    catalog = app.state.catalog
    catalog.refresh_map()
    try:
        vector = app.state.embedder.encode([query])[0]
        return catalog.locate_concept(query, vector)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo ubicar el concepto: {exc}") from exc


@app.post("/images")
def search_by_image(payload: ImageSearchRequest):
    """Ubica cualquier imagen (URL o archivo arrastrado) en el mapa visual SIN persistirla en disco ni BDD."""
    import base64
    import io
    import tempfile
    from PIL import Image
    import requests

    raw_bytes = None
    label = (payload.label or "").strip() or "Imagen consultada"
    if payload.image_data and payload.image_data.startswith("data:image/"):
        try:
            header, b64_part = payload.image_data.split(",", 1)
            raw_bytes = base64.b64decode(b64_part)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Imagen en base64 inválida: {exc}") from exc
    elif payload.image_url:
        url = payload.image_url.strip()
        label = label if payload.label else url.split("/")[-1].split("?")[0][:60] or "Imagen desde URL"
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            )
            resp.raise_for_status()
            if len(resp.content) > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="La imagen supera el límite de 10 MB")
            raw_bytes = resp.content
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"No se pudo descargar la imagen desde la URL: {exc}") from exc
    else:
        raise HTTPException(status_code=422, detail="Envía una URL de imagen o arrastra un archivo de imagen")

    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_resized = img.resize((320, 180), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img_resized.save(buf, format="JPEG", quality=85)
        preview_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "query.jpg"
            tmp_file.write_bytes(buf.getvalue())
            app.state.embedder.enable_images()
            img_vec = app.state.embedder.encode([{"image": [str(tmp_file)]}], batch_size=1, prompt_name=None)[0]

        catalog = app.state.catalog
        catalog.refresh_map()
        return catalog.locate_external_image(img_vec, label=label, preview_url=preview_b64)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo procesar la imagen: {exc}") from exc


@app.post("/profiles")
def analyze_profile(payload: ProfileRequest):
    query = payload.profile.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Ingresa un @handle o link de canal/playlist de YouTube")
    catalog = app.state.catalog
    catalog.refresh_map()
    try:
        res = fetch_profile_videos(query, max_videos=40)
        if res is None:
            raise HTTPException(status_code=404, detail="No se encontró el canal o no tiene videos públicos")
        profile_info, videos = res
        texts = [build_text(v) for v in videos]
        vectors = app.state.embedder.encode(texts, batch_size=32)
        return catalog.locate_profile(profile_info, videos, vectors)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo analizar el perfil: {exc}") from exc


web_dir = Path("web")
if web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="frontend")

