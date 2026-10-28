"""Descarga shards de YouTube-8M usando el script oficial del proyecto."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import requests

DOWNLOAD_SCRIPT = "https://data.yt8m.org/download.py"


def download_shards(output_dir="data/yt8m/video/train", shard="1,100",
                    partition="2/video/train", mirror="us") -> None:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    response = requests.get(DOWNLOAD_SCRIPT, timeout=30)
    response.raise_for_status()
    env = os.environ.copy()
    env.update({"shard": shard, "partition": partition, "mirror": mirror})
    subprocess.run([sys.executable], input=response.content, cwd=destination,
                   env=env, check=True)
