"""Estilo y gráficos que se repiten entre los notebooks."""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from PIL import Image

from .metricas import DATA

INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e6e3"

# un color fijo por representación, el mismo en todos los gráficos
COLORES = {
    "TF-IDF": "#8a8a85",
    "MiniLM": "#b5b5af",
    "Gemma · texto": "#2a78d6",
    "Gemma · imagen": "#eb6834",
    "Gemma · imagen + texto": "#e87ba4",
    "Gemma · fusión tardía": "#1baf7a",
}


def estilo():
    """Fuente con respaldo para títulos en coreano, japonés, ruso, etc."""
    plt.rcParams["font.family"] = ["Seravek", "Arial Unicode MS", "Hiragino Sans", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def thumb(youtube_id: str) -> Image.Image:
    """Miniatura local si existe; si no, la baja de YouTube (para quien no tiene data/thumbs)."""
    path = DATA / "thumbs" / f"{youtube_id}.jpg"
    if path.exists():
        return Image.open(path)
    r = requests.get(f"https://i.ytimg.com/vi/{youtube_id}/mqdefault.jpg", timeout=15)
    return Image.open(io.BytesIO(r.content))


def corto(texto: str, n: int) -> str:
    return texto[:n] + ("…" if len(texto) > n else "")


def plot3d(data: pd.DataFrame, rep: str, titulo: str, color="vertical", leyenda="Categoría", point_size=2):
    """Nube 3D interactiva de una proyección guardada en data/proyecciones.parquet."""
    fig = px.scatter_3d(data, x=f"{rep}_x", y=f"{rep}_y", z=f"{rep}_z",
                        color=color, hover_name="title", opacity=0.6,
                        title=titulo, height=750,
                        color_discrete_sequence=px.colors.qualitative.Alphabet,
                        category_orders={color: data[color].value_counts().index.tolist()})
    fig.update_traces(marker_size=point_size)
    fig.update_layout(legend=dict(itemsizing="constant", font_size=13, title_text=leyenda))
    return fig


def lollipop(metricas: pd.DataFrame, paneles: list[tuple[str, str]], modelos: list[str]):
    """Un panel por métrica; una línea con punta por modelo (más largo = mejor)."""
    fig, axes = plt.subplots(1, len(paneles), figsize=(6 * len(paneles), 0.75 * len(modelos) + 1.8), sharey=True)
    for ax, (col, titulo) in zip(np.atleast_1d(axes), paneles):
        m = metricas.loc[[n for n in modelos if n in metricas.index], col].dropna()
        ypos = np.arange(len(m))[::-1]
        colores = [COLORES.get(n, INK) for n in m.index]
        ax.hlines(ypos, 0, m.values, color=colores, lw=3, zorder=3)
        ax.scatter(m.values, ypos, color=colores, s=90, zorder=4)
        for yy, v in zip(ypos, m.values):
            ax.text(v + m.max() * 0.03, yy, f"{v:.3f}", va="center", fontsize=11, color=INK)
        if col == "knn_acc_original" and "Baseline mayoría" in metricas.index:
            b = metricas.loc["Baseline mayoría", col]
            ax.axvline(b, color=INK, ls="--", lw=1.2)
            ax.text(b, -0.75, f" mayoría {b:.3f}", fontsize=9.5, color=MUTED, va="center")
        ax.set_ylim(-1, len(m) - 0.5)
        ax.set_xlim(0, m.max() * 1.22)
        ax.set_yticks(ypos, m.index, fontsize=11, color=INK)
        ax.set_title(titulo, loc="left", fontsize=13, fontweight="bold", color=INK)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", colors=MUTED)
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    return fig
