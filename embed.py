"""Compatibilidad histórica; la corrida completa parte en `python3 recolect.py`."""

from recolect.embedding import MODEL_NAME, TextEmbedder, build_text, embed_parquet

MODEL = MODEL_NAME
SRC = "data/yt8m_hydrated.parquet"
OUT = "data/emb_text.parquet"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()
    embed_parquet(SRC, OUT, args.limit, args.batch_size, args.max_seq_length)


if __name__ == "__main__":
    main()
