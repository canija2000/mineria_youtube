"""Métricas comunes para comparar las representaciones de los videos.

Todo parte de `data/embeddings_consolidated.parquet`, que escribe `recolect.py`:
una fila por video de la muestra de YT-8M y una columna `emb_<rep>` por
representación. Los notebooks de `notebooks/` usan estas funciones para no
repetir código, y leen los resultados ya calculados por `main()`.

Las métricas son las mismas del informe:
  knn_acc_original   kNN (k=15, coseno) en el espacio original, validación cruzada 5-fold
  knn_acc_umap       kNN (k=15) sobre el UMAP 3D
  NMI_kmeans_umap    K-Means (k = n° de categorías) sobre el UMAP 3D contra la categoría real
  trustworthiness    qué tan fieles son los vecinos del UMAP 3D a los del espacio original

Uso:
    python3 -m analysis.metricas    # escribe data/metricas.csv, data/metricas_proyecciones.csv
                                    # y data/proyecciones.parquet (~7 min)
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# columna emb_<clave> del consolidado -> nombre para tablas y gráficos
REPRESENTACIONES = {
    "tfidf": "TF-IDF",
    "minilm": "MiniLM",
    "text": "Gemma · texto",
    "image": "Gemma · imagen",
    "imgtext": "Gemma · imagen + texto",
    "fusion": "Gemma · fusión tardía",
}
UMAP_PARAMS = dict(n_components=3, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)


def cargar_vocabulario() -> pd.DataFrame:
    """Las 3.806 etiquetas de YT-8M con su embedding (filas en orden posicional 0..n-1)."""
    return pd.read_parquet(DATA / "emb_vocab.parquet").reset_index(drop=True)


def categoria(labels, vertical_de: dict) -> str:
    """Categoría de un video = la Vertical1 más repetida entre sus etiquetas de YT-8M."""
    verticales = [vertical_de.get(int(l)) for l in labels]
    verticales = [v for v in verticales if isinstance(v, str)]
    return Counter(verticales).most_common(1)[0][0] if verticales else "(Unknown)"


def cargar_videos(solo_completos: bool = True) -> pd.DataFrame:
    """Videos del consolidado con su categoría.

    Con `solo_completos=True` deja solo los que tienen las 6 representaciones
    (los que tienen miniatura), para comparar todas sobre la misma muestra.
    """
    df = pd.read_parquet(DATA / "embeddings_consolidated.parquet")
    df = df[df["emb_text"].notna()]
    if solo_completos:
        df = df[df[[f"emb_{r}" for r in REPRESENTACIONES]].notna().all(axis=1)]
    vocab = cargar_vocabulario()
    vertical_de = dict(zip(vocab["Index"].astype(int), vocab["Vertical1"]))
    df = df.reset_index(drop=True)
    df["vertical"] = [categoria(l, vertical_de) for l in df["labels"]]
    return df


def matriz(df: pd.DataFrame, rep: str) -> np.ndarray:
    """Columna emb_<rep> como matriz (n_videos, dim) en float32."""
    return np.stack(df[f"emb_{rep}"].to_numpy()).astype(np.float32)


def muestra(n: int, tamaño: int = 5000) -> np.ndarray:
    """Índices al azar (fijos) para las métricas que comparan todos los pares."""
    return np.random.default_rng(42).choice(n, min(tamaño, n), replace=False)


def evaluar(X: np.ndarray, U: np.ndarray, y: np.ndarray) -> dict:
    """Las métricas para una representación X y su UMAP 3D U."""
    from sklearn.cluster import KMeans
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    k = len(np.unique(y))
    idx = muestra(len(X))
    return {
        "knn_acc_original": cross_val_score(KNeighborsClassifier(15, metric="cosine"), X, y, cv=5).mean(),
        "knn_acc_umap": cross_val_score(KNeighborsClassifier(15), U, y, cv=5).mean(),
        "NMI_kmeans_umap": normalized_mutual_info_score(y, KMeans(k, n_init=5, random_state=42).fit_predict(U)),
        "trustworthiness": trustworthiness(X[idx], U[idx], n_neighbors=15, metric="cosine"),
    }


def evaluar_proyecciones(X: np.ndarray, U: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Compara formas de llevar X a 3D: PCA, UMAP y PCA (90 % de la varianza) + UMAP.

    Silhouette mide si las categorías quedan separadas; se calcula con coseno en el
    espacio original y con distancia euclidiana en las proyecciones.
    """
    import umap
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.dummy import DummyClassifier
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import normalized_mutual_info_score, silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    k = len(np.unique(y))
    idx = muestra(len(X))
    k90 = int(np.argmax(np.cumsum(PCA().fit(X).explained_variance_ratio_) >= 0.9)) + 1
    espacios = {
        "PCA 3D": PCA(n_components=3, random_state=42).fit_transform(X),
        "UMAP 3D": U,
        f"PCA {k90} + UMAP 3D": umap.UMAP(**UMAP_PARAMS).fit_transform(PCA(n_components=k90, random_state=42).fit_transform(X)),
        "Original (768 dims)": X,
    }
    filas = []
    for nombre, Z in espacios.items():
        original = Z is X
        metrica = "cosine" if original else "euclidean"
        filas.append({
            "proyeccion": nombre,
            "trustworthiness": np.nan if original else trustworthiness(X[idx], Z[idx], n_neighbors=15, metric="cosine"),
            "silhouette": silhouette_score(Z[idx], y[idx], metric=metrica),
            "knn_acc": cross_val_score(KNeighborsClassifier(15, metric=metrica), Z, y, cv=5).mean(),
            "NMI_kmeans": normalized_mutual_info_score(y, KMeans(k, n_init=5, random_state=42).fit_predict(Z)),
        })
    filas.append({"proyeccion": "Baseline mayoría",
                  "knn_acc": cross_val_score(DummyClassifier(strategy="most_frequent"), np.zeros((len(y), 1)), y, cv=5).mean()})
    return pd.DataFrame(filas).set_index("proyeccion").round(3)


def zero_shot_topk(X: np.ndarray, L: np.ndarray, k: int = 10, lote: int = 4096) -> np.ndarray:
    """Para cada fila de X, las k etiquetas de L (posiciones) con mayor similitud coseno.

    Se hace por lotes porque la matriz completa 34.771 × 3.806 pesa ~0,5 GB.
    """
    out = np.empty((len(X), k), dtype=np.int64)
    for i in range(0, len(X), lote):
        S = X[i:i + lote] @ L.T
        top = np.argpartition(-S, k, axis=1)[:, :k]
        orden = np.argsort(-np.take_along_axis(S, top, axis=1), axis=1)
        out[i:i + lote] = np.take_along_axis(top, orden, axis=1)
    return out


def hit_at_k(pred_idx: np.ndarray, labels, ks=range(1, 11)) -> pd.Series:
    """Fracción de videos donde alguna de las k primeras etiquetas predichas es real."""
    reales = [set(map(int, l)) for l in labels]
    return pd.Series({k: np.mean([bool(set(p[:k]) & r) for p, r in zip(pred_idx, reales)]) for k in ks})


def main():
    import umap
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import cross_val_score

    df = cargar_videos()
    y = df["vertical"].to_numpy(dtype=str)
    print(f"muestra: {len(df):,} videos, {len(np.unique(y))} categorías")

    proyecciones = df[["pid", "youtube_id", "title", "vertical"]].copy()
    filas = []
    for rep, nombre in REPRESENTACIONES.items():
        X = matriz(df, rep)
        U = umap.UMAP(**UMAP_PARAMS).fit_transform(X)
        if rep == "text":
            evaluar_proyecciones(X, U, y).to_csv(DATA / "metricas_proyecciones.csv")
        proyecciones[[f"{rep}_x", f"{rep}_y", f"{rep}_z"]] = U
        filas.append({"modelo": nombre, **evaluar(X, U, y)})
        print(f"✓ {nombre}")

    filas.append({
        "modelo": "Baseline mayoría",
        "knn_acc_original": cross_val_score(DummyClassifier(strategy="most_frequent"), np.zeros((len(y), 1)), y, cv=5).mean(),
    })
    metricas = pd.DataFrame(filas).set_index("modelo").round(3)
    metricas.to_csv(DATA / "metricas.csv")
    proyecciones.to_parquet(DATA / "proyecciones.parquet", index=False)
    print(f"\n{metricas.to_string()}\n-> data/metricas.csv, data/metricas_proyecciones.csv, data/proyecciones.parquet")


if __name__ == "__main__":
    main()
