"""API HTTP para agregar un video y devolver su posición y vecinos."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from recolect.embedding import TextEmbedder, build_text
from recolect.youtube import fetch_profile_videos, fetch_video
from . import persistencia
from .catalog import VideoCatalog

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
URL_RE = re.compile(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/))([A-Za-z0-9_-]{11})")


class AddVideoRequest(BaseModel):
    video: str = Field(min_length=11, max_length=2048, description="ID o URL de YouTube")
    representation: str = Field(default="text", description="Espacio de representación: text, image, imgtext (fusión temprana) o fusion (fusión tardía)")


class ConceptRequest(BaseModel):
    concept: str = Field(min_length=1, max_length=200, description="Palabra o concepto a ubicar en el mapa 3D")


class ImageSearchRequest(BaseModel):
    image_url: str | None = Field(default=None, max_length=4096, description="URL de imagen externa")
    image_data: str | None = Field(default=None, description="Data URL base64 de imagen arrastrada/subida")
    label: str | None = Field(default=None, max_length=200, description="Nombre o etiqueta de la imagen")


class ProfileRequest(BaseModel):
    profile: str = Field(min_length=2, max_length=2048, description="@handle, URL de canal o playlist de YouTube")
    representation: str = Field(default="fusion", description="Espacio: text, image, imgtext o fusion")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MAPA_DATA=data/dev permite probar el servidor con los datos del modo prueba.
    data_dir = os.getenv("MAPA_DATA", "data")
    persistencia.restaurar(data_dir)   # en Cloud Run: trae los videos recibidos guardados en el bucket
    app.state.catalog = VideoCatalog(data_dir)
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
    """Todos los puntos del mapa vigente (texto, imagen o fusión) para que la web los dibuje."""
    catalog = app.state.catalog
    catalog.refresh_map()
    return catalog.map_payload(representation=rep)


def _save_thumbnail(video_id: str) -> Path:
    """Guarda la miniatura del video en data/thumbs/ (las de los videos ingresados sí se conservan;
    lo que no se guarda son las imágenes sueltas de /images)."""
    import requests
    thumb_dir = persistencia.carpeta_miniaturas(os.getenv("MAPA_DATA", "data"))
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / f"{video_id}.jpg"
    if not thumb_path.exists() or thumb_path.stat().st_size <= 1000:
        resp = requests.get(f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg", timeout=15)
        resp.raise_for_status()
        thumb_path.write_bytes(resp.content)
    return thumb_path


def _save_thumbnails(video_ids: list[str]) -> None:
    """Descarga en paralelo las miniaturas de un lote de videos; si alguna falla, el video igual queda."""
    from concurrent.futures import ThreadPoolExecutor

    def bajar(video_id):
        try:
            _save_thumbnail(video_id)
        except Exception as exc:
            print(f"[miniatura] no se pudo guardar la de {video_id}: {exc}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(bajar, video_ids))


def _register_video(video_id: str, metadata: dict, catalog: VideoCatalog) -> None:
    """Agrega al catálogo un video que llega por primera vez: metadatos, vector de texto y miniatura."""
    if catalog.has(video_id):
        return
    vector = app.state.embedder.encode([build_text(metadata)])[0]
    catalog.add(video_id, metadata, vector)
    persistencia.guardar(os.getenv("MAPA_DATA", "data"))
    try:
        _save_thumbnail(video_id)
    except Exception as exc:  # sin miniatura el video igual queda en el catálogo
        print(f"[miniatura] no se pudo guardar la de {video_id}: {exc}", flush=True)


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _space_vectors(rep: str, videos: list[dict], text_vectors, catalog: VideoCatalog) -> np.ndarray:
    """Vectores de un lote de videos en un mapa de solo videos base (imagen o fusión).

    Se calculan igual que en recolect/modalities.py. Por orden de preferencia, cada vector sale de:
    1. el mapa, si el video es uno de sus videos base;
    2. la SQLite (`space_vectors`), si ya se calculó en una consulta anterior;
    3. el modelo: se baja la miniatura, se embebe y se guarda en la SQLite para la próxima vez.
    La fusión tardía no se guarda: es la suma normalizada del vector de texto y el de imagen.
    """
    space = catalog.spaces[rep]
    kind = "imgtext" if rep == "imgtext" else "image"   # lo que hay que embeber (y guardar)
    ids = [v["youtube_id"] for v in videos]
    out = np.zeros((len(videos), np.asarray(text_vectors).shape[1]), dtype=np.float32)
    faltan = [k for k, video_id in enumerate(ids) if video_id not in space.positions]
    for k, video_id in enumerate(ids):
        if video_id in space.positions:
            out[k] = space.vectors[space.positions[video_id]]
    guardados = catalog.store.space_vectors(kind, [ids[k] for k in faltan]) if faltan else {}
    por_embeber = [k for k in faltan if ids[k] not in guardados]
    if por_embeber:
        _save_thumbnails([ids[k] for k in por_embeber])
        thumbs = {k: _save_thumbnail(ids[k]) for k in por_embeber}   # ya descargadas: solo da la ruta
        embedder = app.state.embedder
        embedder.enable_images()
        if kind == "imgtext":   # fusión temprana: miniatura y texto en una sola pasada del modelo
            entradas = [{"text": f"task: clustering | query: <|image|> {build_text(videos[k])}",
                         "image": [str(thumbs[k])]} for k in por_embeber]
        else:
            entradas = [{"image": [str(thumbs[k])]} for k in por_embeber]
        nuevos = embedder.encode(entradas, batch_size=8, prompt_name=None)
        calculados = {ids[k]: np.asarray(vec, dtype=np.float32) for k, vec in zip(por_embeber, nuevos)}
        catalog.store.add_space_vectors(kind, calculados)
        persistencia.guardar(os.getenv("MAPA_DATA", "data"))
        guardados.update(calculados)
    for k in faltan:
        vec = guardados[ids[k]]
        out[k] = _unit(_unit(text_vectors[k]) + _unit(vec)) if rep == "fusion" else vec
    return out


def _space_vector(rep: str, video_id: str, metadata: dict, catalog: VideoCatalog) -> np.ndarray:
    """Vector de un video en un mapa de solo videos base (ver _space_vectors)."""
    text_vector = catalog.vectors[catalog.positions[video_id]]
    return _space_vectors(rep, [{"youtube_id": video_id, **metadata}], text_vector.reshape(1, -1), catalog)[0]


@app.post("/videos", status_code=201)
def add_video(payload: AddVideoRequest):
    video_id = parse_video_id(payload.video)
    rep = (payload.representation or "text").strip().lower()
    catalog = app.state.catalog
    catalog.refresh_map()

    if rep in catalog.spaces:
        if video_id in catalog.spaces[rep].positions:
            return catalog.describe_static(video_id, rep)
        try:
            # metadatos completos (con tags): del catálogo de recibidos si ya llegó antes, si no de la API
            metadata = catalog.store.get(video_id) or fetch_video(video_id)
            if metadata is None:
                raise HTTPException(status_code=404, detail="El video no existe o no está disponible")
            _register_video(video_id, metadata, catalog)   # queda guardado igual que en modo texto
            vector = _space_vector(rep, video_id, metadata, catalog)
            return catalog.describe_static(video_id, rep, metadata=metadata, vector=vector)
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
        # Los videos del canal ya están descargados y con vector: quedan guardados como recibidos.
        nuevos = catalog.add_many([(v["youtube_id"], v, vec) for v, vec in zip(videos, vectors)])
        if nuevos:
            persistencia.guardar(os.getenv("MAPA_DATA", "data"))
            _save_thumbnails(nuevos)
        rep = (payload.representation or "text").strip().lower()
        if rep in catalog.spaces and catalog.spaces[rep].ready:
            space_vectors = _space_vectors(rep, videos, vectors, catalog)
            result = catalog.locate_profile(profile_info, videos, space_vectors, space=rep)
        else:
            result = catalog.locate_profile(profile_info, videos, vectors)
        result["profile"]["added_videos"] = len(nuevos)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo analizar el perfil: {exc}") from exc


web_dir = Path("web")
if web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="frontend")

