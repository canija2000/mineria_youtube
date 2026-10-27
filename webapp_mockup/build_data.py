"""Datos para el mockup de la web app (solo front, para las diapositivas de la propuesta).

Genera webapp_mockup/data.js con:
  - una muestra de la proyección UMAP 3D de EmbeddingGemma 2 (texto), con título y categoría;
  - un video de ejemplo con sus vecinos más cercanos en el espacio original (768-d);
  - 3 frames del video (los que YouTube genera al 25/50/75 %) con sus palabras más cercanas
    del vocabulario de YT-8M (zero-shot con EmbeddingGemma 2);
  - un "perfil" de ejemplo: 40 videos que simulan los últimos likes de un usuario.

Uso:
  python3 webapp_mockup/build_data.py
"""

import json
import os
import socket

import numpy as np
import pandas as pd
import requests

# La red local tiene IPv6 roto: forzar IPv4.
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, *a, **k: _orig(h, p, socket.AF_INET, *a, **k)

OUT = "webapp_mockup"
DEMO = "W-MuJ9tlwPA"  # Porsche 911 GT3 RS vs Lamborghini Gallardo
N_POINTS = 8000
SEED = 7

proj = pd.read_parquet("data/projections.parquet",
                       columns=["youtube_id", "title", "vertical", "umap_x", "umap_y", "umap_z"])
meta = pd.read_parquet("data/yt8m_hydrated.parquet",
                       columns=["youtube_id", "channel_title", "view_count", "duration"])
proj = proj.merge(meta, on="youtube_id", how="left")
emb = pd.read_parquet("data/emb_text.parquet")
X = np.stack(emb["emb"].to_numpy())
row = {y: i for i, y in enumerate(emb["youtube_id"])}

# --- vecinos reales del video de ejemplo (coseno en 768-d)
sims = X @ X[row[DEMO]]
order = [emb["youtube_id"][i] for i in np.argsort(-sims)[1:13]]

# --- perfil de ejemplo: 40 videos de autos y juegos de carreras/acción
rng = np.random.default_rng(SEED)
autos = proj[proj.vertical == "Autos & Vehicles"].sample(26, random_state=SEED)
games = proj[(proj.vertical == "Games") & proj.title.str.contains("GTA|Forza|Need for Speed|Gran Turismo", case=False, na=False)]
games = games.sample(min(14, len(games)), random_state=SEED)
profile = pd.concat([autos, games])["youtube_id"].tolist()

# lejos del centro del perfil = "fuera de tu burbuja"
c = X[[row[y] for y in profile]].mean(0)
c /= np.linalg.norm(c)
far_idx = np.argsort(X @ c)[:400]
far = proj.set_index("youtube_id").loc[[emb["youtube_id"][i] for i in far_idx]]
far = far.groupby("vertical").head(1).head(5).index.tolist()

keep = set([DEMO] + order + profile + far)
sample = pd.concat([proj[proj.youtube_id.isin(keep)],
                    proj[~proj.youtube_id.isin(keep)].sample(N_POINTS - len(keep), random_state=SEED)])

# --- frames del video de ejemplo + palabras más cercanas del vocabulario
os.makedirs(f"{OUT}/frames", exist_ok=True)
paths = []
for k in (1, 2, 3):
    p = f"{OUT}/frames/{DEMO}_{k}.jpg"
    if not os.path.exists(p):
        r = requests.get(f"https://i.ytimg.com/vi/{DEMO}/{k}.jpg", timeout=15)
        r.raise_for_status()
        open(p, "wb").write(r.content)
    paths.append(p)
thumb = f"{OUT}/frames/{DEMO}_thumb.jpg"
if not os.path.exists(thumb):
    open(thumb, "wb").write(requests.get(f"https://i.ytimg.com/vi/{DEMO}/mqdefault.jpg", timeout=15).content)

from multimodal import encode, load_model  # noqa: E402  (carga pesada, solo aquí)

vocab = pd.read_parquet("data/emb_vocab.parquet")
names = vocab["Name"].to_numpy()  # por posición: el índice del parquet no es 0..n-1
L = np.stack(vocab["emb"].to_numpy()).astype(np.float32)
model = load_model()
F = encode(model, [{"image": [p]} for p in paths + [thumb]], "frames")
S = F @ L.T
words = [[{"w": names[j], "s": round(float(S[i, j]), 3)} for j in np.argsort(-S[i])[:5]]
         for i in range(len(F))]

pts = [[round(float(r.umap_x), 3), round(float(r.umap_y), 3), round(float(r.umap_z), 3)]
       for r in sample.itertuples()]
cats = sample["vertical"].value_counts().index.tolist()
info = {r.youtube_id: {"t": r.title, "ch": r.channel_title, "v": r.view_count, "d": r.duration, "c": r.vertical}
        for r in sample.itertuples() if r.youtube_id in keep}

data = {
    "cats": cats,
    "ids": sample["youtube_id"].tolist(),
    "titles": sample["title"].fillna("").str.slice(0, 90).tolist(),
    "cat": [cats.index(v) for v in sample["vertical"]],
    "pts": pts,
    "demo": DEMO,
    "neighbors": order,
    "neighbor_sims": [round(float(s), 3) for s in np.sort(-sims)[1:13] * -1],
    "profile": profile,
    "far": far,
    "frames": [{"src": f"frames/{DEMO}_{k}.jpg", "pct": pct, "words": words[i]}
               for i, (k, pct) in enumerate([(1, 25), (2, 50), (3, 75)])],
    "thumb_words": words[3],
    "info": info,
}
with open(f"{OUT}/data.js", "w") as f:
    f.write("window.DATA = " + json.dumps(data, ensure_ascii=False) + ";\n")
print("ok", len(pts), "puntos;", "perfil", len(profile), "; lejos", far)
print("frames:", [[w["w"] for w in fr] for fr in words])
