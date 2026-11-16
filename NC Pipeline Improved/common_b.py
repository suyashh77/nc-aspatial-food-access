"""Shared helpers for Pipeline B (improvement experiments). Reads Pipeline A's outputs; never writes to them."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parent / "NC Pipeline"
sys.path.insert(0, str(A))
sys.path.insert(0, str(A / "decisions"))
import numpy as np
import pandas as pd

import config as C
from definitions import DIMS, embed_definitions
import step4_match as S4

OUT = HERE / "out"


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def umap5(X, fit_rows=None):
    from cuml.manifold import UMAP
    m = UMAP(**C.UMAP_PARAMS)
    if fit_rows is None:
        return np.asarray(m.fit_transform(X), dtype=np.float32)
    m.fit(X[fit_rows])
    out = np.empty((len(X), 5), dtype=np.float32)
    for s in range(0, len(X), 500_000):
        out[s:s + 500_000] = np.asarray(m.transform(X[s:s + 500_000]))
    return out


def hdbscan(u5, mcs):
    from cuml.cluster import HDBSCAN
    p = dict(min_cluster_size=mcs, **C.HDBSCAN_FIXED)
    p.pop("prediction_data")
    return np.asarray(HDBSCAN(**p).fit(u5).labels_).astype(int)


def transform_assign_from_labels(E, labels):
    """BERTopic-style: topic embedding = mean member embedding (incl. outlier topic), then every row goes to the
    most similar topic embedding (what Teresa's saved-model transform does)."""
    ids = np.unique(labels)
    pos = {t: i for i, t in enumerate(ids)}
    li = np.array([pos[t] for t in labels])
    acc = np.zeros((len(ids), E.shape[1]))
    np.add.at(acc, li, E)
    te = unit(acc)
    best = np.empty(len(E), dtype=int)
    for s in range(0, len(E), 200_000):
        best[s:s + 200_000] = ids[(E[s:s + 200_000] @ te.T).argmax(1)]
    return best


def centroids(E, labels):
    ids = np.array(sorted(t for t in np.unique(labels) if t >= 0))
    pos = {t: i for i, t in enumerate(ids)}
    keep = labels >= 0
    acc = np.zeros((len(ids), E.shape[1]))
    np.add.at(acc, np.array([pos[t] for t in labels[keep]]), E[keep])
    return ids, unit(acc)


def top_words(texts, labels, topn=8, sample=300_000, seed=0):
    """Quick class-based TF-IDF (BERTopic's c-TF-IDF formula) on a sample, for describing topics."""
    from sklearn.feature_extraction.text import CountVectorizer
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), min(sample, len(texts)), replace=False)
    t = pd.Series(np.asarray(texts, dtype=object)[idx])
    lab = labels[idx]
    cv = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2), max_features=60_000)
    X = cv.fit_transform(t)
    ids = np.array(sorted(set(lab) - {-1}))
    import scipy.sparse as sp
    M = np.vstack([np.asarray(X[lab == k].sum(0)).ravel() for k in ids]).astype(float)
    tf = M / (M.sum(1, keepdims=True) + 1e-12)
    idf = np.log(1 + M.sum(1).mean() / (M.sum(0) + 1))
    sc = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    return {int(k): list(vocab[np.argsort(-sc[i])[:topn]]) for i, k in enumerate(ids)}


def generic_topics(words: dict, share=0.7) -> set:
    g = C.GENERIC_WORDS
    return {k for k, ws in words.items() if np.mean([all(p in g for p in w.split()) for w in ws]) >= share}


def stars_eta2(labels, stars):
    """Share of star variance explained by topic membership (0 = topics say nothing about mood)."""
    df = pd.DataFrame({"t": labels, "y": stars}).dropna()
    df = df[df.t >= 0]
    grand = df.y.mean()
    between = df.groupby("t").y.agg(["mean", "size"]).pipe(lambda g: (g["size"] * (g["mean"] - grand) ** 2).sum())
    return round(float(between / ((df.y - grand) ** 2).sum()), 4)


def cashier_check(texts, labels):
    """Reviews about rude vs friendly cashiers: do they land in the same topic?"""
    t = pd.Series(np.asarray(texts, dtype=object)).str.lower()
    rude = t.str.contains(r"\brude\b") & t.str.contains(r"cashier")
    nice = t.str.contains(r"\b(?:friendly|nice|kind|sweet)\b") & t.str.contains(r"cashier") & ~t.str.contains(r"\brude\b")
    lab = pd.Series(labels)
    r, n = lab[rude.to_numpy()].value_counts(normalize=True), lab[nice.to_numpy()].value_counts(normalize=True)
    overlap = float(sum(min(r.get(k, 0), n.get(k, 0)) for k in set(r.index) | set(n.index)))
    return {"rude_cashier_reviews": int(rude.sum()), "friendly_cashier_reviews": int(nice.sum()),
            "top_topic_rude": int(r.index[0]) if len(r) else None, "top_topic_friendly": int(n.index[0]) if len(n) else None,
            "distribution_overlap": round(overlap, 3)}


def save(name, R):
    (OUT / f"{name}.json").write_text(json.dumps(R, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
