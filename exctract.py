import tensorflow as tf
import pandas as pd
import numpy as np
import glob

rows = []
for raw in tf.data.TFRecordDataset(glob.glob("data/yt8m/video/train/*.tfrecord")):
    f = tf.train.Example.FromString(raw.numpy()).features.feature
    rows.append(
        {
            "pid": f["id"].bytes_list.value[0].decode(),
            "labels": list(f["labels"].int64_list.value),
            "mean_rgb": np.array(f["mean_rgb"].float_list.value, dtype=np.float32),
            "mean_audio": np.array(f["mean_audio"].float_list.value, dtype=np.float32),
        }
    )
pd.DataFrame(rows).to_parquet("data/yt8m_sample.parquet")
