"""Step 1: clean and embed every NC review that has text.

    .venv/bin/python step1_embed.py            # ~10 min on the RTX A4500

Reads the master review DB read-only. Writes
  data/step1_reviews.parquet     one row per embedded review: id (DB row id), store_id,
                                 review_id, rating, date_relative, scraped_at, text, processed_text
  data/step1_embeddings.npy      float32 (n, 384), row-aligned with the parquet
  data/step1_meta.json           counts, DB checksum, model, timings
Reviews whose cleaned text is empty ("cleaned out of existence", Teresa notebook 3 cell 8) are
counted and dropped, as in the pilot.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time

import numpy as np
import pandas as pd

import config as C

CHUNK = 50_000


def clean_text(text, custom_stopwords=C.STOPWORDS):
    """
    Function to clean text data by removing html tags, punctuation, and lowercasing the text.
    (Teresa notebook 2 cell 6, verbatim; the stopword list is a parameter.)
    """
    #lowercase text
    text = text.lower()

    #remove html tags
    text = re.sub(r'<.*?>', '', text)

    #remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    #remove custom stopwords
    text = ' '.join(word for word in text.split() if word not in custom_stopwords)

    return text


def sha256(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_reviews() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{C.REVIEWS_DB}?mode=ro", uri=True)
    df = pd.read_sql("select id, store_id, review_id, rating, date_relative, scraped_at, "
                     "review_text as text from reviews "
                     "where review_text is not null and trim(review_text) <> '' order by id", con)
    con.close()
    return df


def main() -> None:
    from sentence_transformers import SentenceTransformer
    C.DATA.mkdir(exist_ok=True)
    t0 = time.time()
    df = load_reviews()
    n_text = len(df)
    df["processed_text"] = df["text"].map(clean_text)
    keep = df["processed_text"].str.strip().ne("")
    df = df[keep].reset_index(drop=True)
    print(f"{n_text:,} reviews with text; {int((~keep).sum()):,} cleaned out of existence; "
          f"{len(df):,} to embed", flush=True)
    df.to_parquet(C.DATA / "step1_reviews.parquet", index=False)
    model = SentenceTransformer(C.EMBEDDING_MODEL, device="cuda")
    dim = model.get_sentence_embedding_dimension()
    # Written chunk by chunk into a memory-mapped .npy: the shared VM often has <5 GB free RAM.
    emb = np.lib.format.open_memmap(C.DATA / "step1_embeddings.npy", mode="w+", dtype=np.float32,
                                    shape=(len(df), dim))
    texts = df["processed_text"].tolist()
    t1 = time.time()
    for start in range(0, len(texts), CHUNK):
        emb[start:start + CHUNK] = model.encode(texts[start:start + CHUNK], batch_size=C.EMBED_BATCH,
                                                convert_to_numpy=True)
        print(f"  embedded {min(start + CHUNK, len(texts)):,} / {len(texts):,} "
              f"({time.time() - t1:.0f}s)", flush=True)
    emb.flush()
    t_embed = time.time() - t1
    meta = {"reviews_db": str(C.REVIEWS_DB), "reviews_db_sha256": sha256(C.REVIEWS_DB),
            "reviews_with_text": n_text, "cleaned_out": int((~keep).sum()), "embedded": len(df),
            "stores": int(df.store_id.nunique()), "model": C.EMBEDDING_MODEL,
            "max_seq_length": model.max_seq_length, "dim": int(emb.shape[1]),
            "norm_mean": float(np.linalg.norm(emb[:10_000], axis=1).mean()),
            "stopwords_teresa": len(C.TERESA_STOPWORDS), "stopwords_nc": C.NC_STOPWORDS,
            "embed_seconds": round(t_embed), "total_seconds": round(time.time() - t0)}
    (C.DATA / "step1_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
