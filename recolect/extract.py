"""Extrae registros de TFRecord de YouTube-8M y los guarda en Parquet."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def extract_tfrecords(pattern: str, output: str | Path) -> pd.DataFrame:
    import tensorflow as tf

    files = [Path(path) for path in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No hay archivos TFRecord que coincidan con: {pattern}")

    rows = []
    for raw in tf.data.TFRecordDataset([str(path) for path in files]):
        features = tf.train.Example.FromString(raw.numpy()).features.feature
        rows.append({
            "pid": features["id"].bytes_list.value[0].decode(),
            "labels": list(features["labels"].int64_list.value),
            "mean_rgb": np.asarray(features["mean_rgb"].float_list.value, dtype=np.float32),
            "mean_audio": np.asarray(features["mean_audio"].float_list.value, dtype=np.float32),
        })

    frame = pd.DataFrame(rows)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/yt8m/video/train/*.tfrecord")
    parser.add_argument("--output", default="data/yt8m_sample.parquet")
    args = parser.parse_args()
    result = extract_tfrecords(args.input, args.output)
    print(f"{len(result):,} videos -> {args.output}")


if __name__ == "__main__":
    main()
