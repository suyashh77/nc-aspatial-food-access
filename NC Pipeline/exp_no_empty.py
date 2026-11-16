"""Experiment: does removing reviews with no specific words give more useful topics?

    .venv/bin/python exp_no_empty.py          # ~20 min on the GPU

Keeps only reviews with at least one word outside config.GENERIC_WORDS and English stop words, redoes
UMAP and HDBSCAN on them with the same settings, and compares with the current model (data/step3_*).
Writes only to data/exp_no_empty/; the current model is not touched.

Topic words: c-TF-IDF as in step 3 for the metrics, plus a second list with praise-only phrases hidden.
"""
from __future__ import annotations

import ast
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config as C

OUT = C.DATA / "exp_no_empty"
SIZES = [2000, 1000, 4000]
TOKEN = re.compile(r"(?u)\b\w\w+\b")
EMPTY_WORDS = ENGLISH_STOP_WORDS | C.GENERIC_WORDS


def count_specific(text: str) -> int:
    """Number of words that aren't stop words or generic praise / filler."""
    return sum(1 for t in TOKEN.findall(text.lower()) if t not in EMPTY_WORDS)


def is_generic_phrase(phrase: str) -> bool:
    return count_specific(phrase) == 0


def generic_frac(words: list) -> float:
    return float(np.mean([is_generic_phrase(w) for w in words])) if words else 1.0


def hidden_top_words(model, k: int = 10) -> dict:
    """Top c-TF-IDF terms per topic after dropping phrases made only of generic words."""
    names = model.vectorizer_model.get_feature_names_out()
    keep = np.array([not is_generic_phrase(n) for n in names])
    M = model.c_tf_idf_.tocsr()
    out = {}
    for row, t in enumerate(sorted(set(model.topics_))):
        scores = np.asarray(M[row].todense()).ravel() * keep
        top = np.argsort(-scores)[:k]
        out[int(t)] = [names[i] for i in top if scores[i] > 0]
    return out


def compare(base: np.ndarray, new: np.ndarray) -> pd.DataFrame:
    """For each current topic, the new topic that overlaps it most (Jaccard, on kept reviews)."""
    rows = []
    b_sizes = pd.Series(base).value_counts()
    n_sizes = pd.Series(new).value_counts()
    tab = pd.DataFrame({"b": base, "n": new}).query("b >= 0 and n >= 0").value_counts()
    for t in sorted(b_sizes.index[b_sizes.index >= 0]):
        best, bj = -1, 0.0
        if t in tab.index.get_level_values("b"):
            for n, c in tab.loc[t].items():
                j = c / (b_sizes[t] + n_sizes[n] - c)
                if j > bj:
                    best, bj = int(n), float(j)
        rows.append({"current_topic": int(t), "best_new_topic": best, "jaccard": round(bj, 3)})
    return pd.DataFrame(rows)


def main() -> None:
    from step2_umap import make_umap
    from step3_topics import cluster, fit_topics, label_stats, quality
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id", "processed_text"])
    base = pd.read_parquet(C.DATA / "step3_review_topics.parquet", columns=["id", "topic"])
    assert (rv.id.values == base.id.values).all()
    keep = rv.processed_text.map(count_specific).to_numpy() > 0
    np.save(OUT / "keep_mask.npy", keep)
    docs = rv.processed_text[keep].tolist()
    print(f"kept {keep.sum():,} of {len(keep):,}; removed {(~keep).sum():,} ({(~keep).mean():.1%})", flush=True)
    emb = np.array(np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")[keep], dtype=np.float32)
    u5_f = OUT / "umap5.npy"
    if u5_f.exists():
        u5 = np.load(u5_f)
    else:
        u5 = np.asarray(make_umap(C.UMAP_PARAMS).fit_transform(emb), dtype=np.float32)
        np.save(u5_f, u5)
    print(f"umap done {time.time() - t0:.0f}s", flush=True)

    info = pd.read_csv(C.DATA / "step3_topic_info.csv")
    info = info[info.Topic >= 0]
    cur_words = {int(r.Topic): ast.literal_eval(r.Representation) for r in info.itertuples()}
    cur_generic = sum(generic_frac(w) >= 0.5 for w in cur_words.values())
    base_kept = base.topic.to_numpy()[keep]
    cur = json.loads((C.DATA / "step3_meta.json").read_text())
    results = [{"model": "current", "min_cluster_size": cur["min_cluster_size"], "reviews": len(keep), "n_topics": cur["n_topics"],
                "outlier_share": cur["outlier_share"], "outlier_share_on_kept": round(float((base_kept < 0).mean()), 4),
                "coherence_cv": cur["coherence_cv"], "topic_diversity": cur["topic_diversity"], "generic_topics": int(cur_generic)}]
    for mcs in SIZES:
        t1 = time.time()
        labels, probs = cluster(u5, mcs)
        model, _ = fit_topics(docs, emb, labels, representations=False)
        topics = np.asarray(model.topics_)
        np.save(OUT / f"topics_mcs{mcs}.npy", topics)
        np.save(OUT / f"probs_mcs{mcs}.npy", probs)
        words = {t: [w for w, _ in ws] for t, ws in model.get_topics().items() if t != -1}
        hidden = hidden_top_words(model)
        stats = label_stats(topics)
        row = {"model": "no_empty", "min_cluster_size": mcs, "reviews": len(docs), **{k: stats[k] for k in ("n_topics", "outlier_share")},
               "outlier_share_on_kept": stats["outlier_share"], **quality(model, docs),
               "generic_topics": int(sum(generic_frac(w) >= 0.5 for w in words.values())), "seconds": round(time.time() - t1)}
        results.append(row)
        sizes = pd.Series(topics).value_counts()
        pd.DataFrame([{"topic": t, "reviews": int(sizes[t]), "words": ", ".join(words[t]), "words_praise_hidden": ", ".join(hidden.get(t, [])),
                       "generic_frac": round(generic_frac(words[t]), 2)} for t in sorted(words)]).to_csv(OUT / f"topic_words_mcs{mcs}.csv", index=False)
        compare(base_kept, topics).to_csv(OUT / f"match_current_mcs{mcs}.csv", index=False)
        print(json.dumps(row), flush=True)
        pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
