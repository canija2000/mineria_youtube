"""Evalúa las 6 representaciones sobre la MISMA muestra de videos de multimodal.py.

Representaciones:
  TF-IDF, MiniLM, Texto (Gemma2)   -> embeddings de texto ya calculados (filtrados a la muestra)
  Imagen (Gemma2), Img+texto        -> de multimodal.py
  Fusión texto+imagen               -> normalize(texto + imagen), se calcula aquí

Métricas (las mismas de siempre):
  knn_acc_original   kNN (k=15, coseno) sobre el espacio original, 5-fold CV
  knn_acc_umap       kNN sobre UMAP 3D
  NMI_kmeans_umap    K-Means (k = n° de categorías) sobre UMAP 3D vs categoría real
  + baseline de mayoría y hit@k del zero-shot

Salida: data/metricas_multimodal.csv

Uso:
  python3 evaluate_multimodal.py
"""

import duckdb
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

VOCAB_URL = "https://research.google.com/youtube8m/csv/2/vocabulary.csv"
OUT = "data/metricas_multimodal.csv"

EMBEDDINGS = {
    "TF-IDF": "data/emb_tfidf.parquet",
    "MiniLM": "data/emb_minilm.parquet",
    "Texto (Gemma2)": "data/emb_text.parquet",
    "Imagen (Gemma2)": "data/emb_image.parquet",
    "Img+texto (Gemma2)": "data/emb_imgtext.parquet",
}


def load_sample():
    """La muestra = los videos que tienen embedding de imagen. Agrega categoría y etiquetas reales."""
    base = pd.read_parquet("data/emb_image.parquet", columns=["pid"])
    cat = duckdb.sql(f"""
        SELECT s.pid, mode(v.Vertical1) AS vertical, list(s.lbl) AS labels
        FROM (SELECT pid, unnest(labels) AS lbl FROM 'data/yt8m_sample.parquet') s
        JOIN read_csv('{VOCAB_URL}') v ON v.Index = s.lbl
        GROUP BY s.pid
    """).df()
    base = base.merge(cat, on="pid", how="left")
    base["vertical"] = base["vertical"].fillna("(Unknown)")
    return base


def load_aligned(base, path):
    """Embeddings de `path` en el mismo orden de filas que `base` (unidos por pid)."""
    emb = base[["pid"]].merge(pd.read_parquet(path, columns=["pid", "emb"]), on="pid", how="left")["emb"]
    assert emb.notna().all(), f"faltan videos de la muestra en {path}"
    return np.stack(emb).astype(np.float32)


def main():
    base = load_sample()
    y = base["vertical"].to_numpy(dtype=str)
    k = len(np.unique(y))
    print(f"muestra: {len(base)} videos, {k} categorías")

    reps = {name: load_aligned(base, path) for name, path in EMBEDDINGS.items()}
    reps["Fusión texto+imagen"] = normalize(reps["Texto (Gemma2)"] + reps["Imagen (Gemma2)"])

    params = dict(n_components=3, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
    rows = []
    for name, X in reps.items():
        U = umap.UMAP(**params).fit_transform(X)
        rows.append({
            "modelo": name,
            "knn_acc_original": cross_val_score(KNeighborsClassifier(15, metric="cosine"), X, y, cv=5).mean(),
            "knn_acc_umap": cross_val_score(KNeighborsClassifier(15), U, y, cv=5).mean(),
            "NMI_kmeans_umap": normalized_mutual_info_score(y, KMeans(k, n_init=5, random_state=42).fit_predict(U)),
        })
        print(f"✓ {name}")
    rows.append({
        "modelo": "baseline mayoría",
        "knn_acc_original": cross_val_score(DummyClassifier(strategy="most_frequent"), reps["TF-IDF"], y, cv=5).mean(),
    })

    metricas = pd.DataFrame(rows).set_index("modelo").round(3)
    metricas.to_csv(OUT)
    print(f"\n{metricas.to_string()}\n-> {OUT}")

    # zero-shot: ¿alguna de las k etiquetas predichas desde la miniatura es una etiqueta real?
    zs = pd.read_parquet("data/zeroshot_image.parquet", columns=["pid", "pred_idx"]).merge(base[["pid", "labels"]], on="pid")
    for kk in (1, 5, 10):
        hit = np.mean([len(set(p[:kk]) & set(l)) > 0 for p, l in zip(zs["pred_idx"], zs["labels"])])
        print(f"zero-shot imagen hit@{kk}: {hit:.3f}")


if __name__ == "__main__":
    main()
