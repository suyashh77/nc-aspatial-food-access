"""Step 3: cluster the reviews into topics with HDBSCAN and build BERTopic topics
(Teresa notebook 2, cells 16-48).

    .venv/bin/python step3_topics.py grid          # HDBSCAN cluster-size grid + quality metrics
    .venv/bin/python step3_topics.py fit --mcs N   # final NC model (KeyBERT + MMR representations)
    .venv/bin/python step3_topics.py frozen        # Raleigh arm: Teresa's saved model applied to NC

How this maps onto Teresa's notebook:
  * UMAP and HDBSCAN run outside BERTopic (step2_umap.py, `cluster` below) and are handed to it
    with BaseDimensionalityReduction / BaseCluster ("manual BERTopic"). Same algorithms and
    parameters, but the 5-D coordinates and cluster labels are saved and can be re-used.
  * Keywords: the CountVectorizer settings are hers (English stop words, 1-3-grams, min_df=2 across
    topics). At 975K reviews sklearn's analyzer would hold the outlier topic's ~10M n-grams in
    memory, so `StreamingNgrams` yields the same n-grams lazily, and the vocabulary is fixed from a
    200K-review sample (n-grams seen >= 3 times there), then filtered to n-grams in >= 2 topics.
    Teresa's note: this step only changes how topics are described, not the clustering.
  * Quality metrics are her functions, verbatim: C_v coherence (on a 100K-review sample; the full
    corpus is too slow), topic diversity and per-topic uniqueness.

Outputs (data/):
  step3_grid/labels_mcs<N>.npy, step3_grid/probs_mcs<N>.npy, step3_grid.csv
  step3_bertopic_model/           saved like Teresa's (safetensors + c-TF-IDF + embedding model name)
  step3_topic_info.csv            Topic, Count, Name, Representation, KeyBERT, MMR, Representative_Docs
  step3_review_topics.parquet     id, store_id, review_id, topic, topic_prob (+ frozen_* after `frozen`)
  step3_meta.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

import config as C

GRID = C.DATA / "step3_grid"
TOKEN = re.compile(r"(?u)\b\w\w+\b")          # sklearn's default token_pattern


# ------------------------------- n-grams ----------------------------------------
class StreamingNgrams:
    """sklearn's word analyzer (lowercase, default token pattern, English stop words removed,
    then 1..n-grams), yielded lazily so a topic's concatenated text is never held as a list."""

    def __init__(self, n: int = 3):
        self.n = n

    def __call__(self, doc: str):
        window: deque = deque(maxlen=self.n)
        for m in TOKEN.finditer(doc.lower()):
            tok = m.group()
            if tok in ENGLISH_STOP_WORDS:
                continue
            window.append(tok)
            w = list(window)
            for k in range(1, len(w) + 1):
                yield " ".join(w[-k:])


def sample_vocabulary(docs: list, n: int = 200_000, min_count: int = 3) -> list:
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(docs), size=min(n, len(docs)), replace=False)
    cv = CountVectorizer(analyzer=StreamingNgrams(C.VECTORIZER_PARAMS["ngram_range"][1]))
    X = cv.fit_transform(docs[i] for i in idx)
    counts = np.asarray(X.sum(axis=0)).ravel()
    names = cv.get_feature_names_out()
    return sorted(names[counts >= min_count].tolist())


def topic_vocabulary(docs: list, labels: np.ndarray, vocab: list) -> list:
    """Keep n-grams that occur in >= min_df topics (Teresa's min_df=2, applied to topic documents)."""
    from scipy import sparse
    cv = CountVectorizer(analyzer=StreamingNgrams(C.VECTORIZER_PARAMS["ngram_range"][1]), vocabulary=vocab)
    X = cv.transform(docs)                                     # one row per review (streamed)
    topics, inv = np.unique(labels, return_inverse=True)
    T = sparse.csr_matrix((np.ones(len(labels)), (inv, np.arange(len(labels)))), shape=(len(topics), len(labels)))
    df = np.asarray(((T @ X) > 0).sum(axis=0)).ravel()
    return [v for v, d in zip(vocab, df) if d >= C.VECTORIZER_PARAMS["min_df"]]


# ------------------------------- Teresa's metrics (verbatim) -------------------
def extract_top_words_from_bertopic(model, topn=10, drop_outlier=True):
    """
    Pulls the top-N c-TF-IDF words per topic from a BERTopic model.
    Returns:
        topic_words:  {topic_id: [w1,...,wN]}
    """
    topics = model.get_topics()  # {topic_id: [(word, score), ...]}
    topic_words = {}
    for k, ws in topics.items():
        if drop_outlier and k == -1:
            continue
        if not ws:
            continue
        topic_words[k] = [w for w, _ in ws[:topn]]
    return topic_words


def compute_cv_coherence(topic_words_dict, texts_tokenized, *, window_size=110, processes=8):
    """C_v coherence per topic and overall (Teresa notebook 2 cell 40; `processes` added so gensim
    doesn't start one worker per core on the shared VM)."""
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    topic_ids = sorted(topic_words_dict.keys())
    topics_list = [topic_words_dict[k] for k in topic_ids]
    dictionary = Dictionary(texts_tokenized)
    cm = CoherenceModel(topics=topics_list, texts=texts_tokenized, dictionary=dictionary,
                        coherence="c_v", window_size=window_size, processes=processes)
    per_topic = cm.get_coherence_per_topic()
    overall = float(np.mean(per_topic)) if per_topic else float("nan")
    return {"per_topic": dict(zip(topic_ids, per_topic)), "overall": overall}


def topic_uniqueness_and_diversity(topic_words_dict, topn=None):
    """
    Returns:
        per_topic_uniqueness: {topic_id: TU_k}
        model_topic_diversity: float in [0,1]
    """
    if topn is not None:
        tw = {k: v[:topn] for k, v in topic_words_dict.items()}
        N = topn
    else:
        minN = min(len(v) for v in topic_words_dict.values())
        tw = {k: v[:minN] for k, v in topic_words_dict.items()}
        N = minN
    word_topics = defaultdict(set)
    for k, words in tw.items():
        for w in set(words):
            word_topics[w].add(k)
    per_topic_uniqueness = {}
    for k, words in tw.items():
        uniq = sum(1 for w in words if len(word_topics[w]) == 1)
        per_topic_uniqueness[k] = uniq / float(N)
    total = len(tw) * N
    unique_word_count = sum(1 for w, ts in word_topics.items() if len(ts) >= 1)
    model_topic_diversity = unique_word_count / float(total)
    return per_topic_uniqueness, model_topic_diversity


# ------------------------------- pipeline pieces --------------------------------
def load_inputs():
    reviews = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id", "store_id", "review_id", "processed_text"])
    emb = np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")
    u5 = np.load(C.DATA / "step2_umap5.npy")
    assert len(reviews) == len(emb) == len(u5), "step 1 / step 2 outputs are not row-aligned"
    return reviews, emb, u5


def cluster(u5: np.ndarray, mcs: int) -> tuple:
    """HDBSCAN on the 5-D UMAP space -> (labels, membership probabilities)."""
    params = dict(min_cluster_size=mcs, **C.HDBSCAN_FIXED)
    if C.ENGINE == "cuml":
        from cuml.cluster import HDBSCAN
        params.pop("prediction_data")
        m = HDBSCAN(**params).fit(u5)
    else:
        from hdbscan import HDBSCAN
        m = HDBSCAN(core_dist_n_jobs=8, **params).fit(u5)
    return np.asarray(m.labels_).astype(int), np.asarray(m.probabilities_).astype(np.float32)


def bertopic_model(vocab: list, representations: bool):
    from bertopic import BERTopic
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from sentence_transformers import SentenceTransformer
    rep = {"KeyBERT": KeyBERTInspired(), "MMR": MaximalMarginalRelevance(diversity=C.MMR_DIVERSITY)} if representations else None
    return BERTopic(
        embedding_model=SentenceTransformer(C.EMBEDDING_MODEL, device="cuda"),
        umap_model=BaseDimensionalityReduction(),          # step 2 output, already reduced
        hdbscan_model=BaseCluster(),                       # labels from `cluster`
        vectorizer_model=CountVectorizer(analyzer=StreamingNgrams(C.VECTORIZER_PARAMS["ngram_range"][1]), vocabulary=vocab),
        representation_model=rep,
        top_n_words=C.TOP_N_WORDS,
        verbose=True,
    )


def fit_topics(docs: list, emb, labels: np.ndarray, representations: bool):
    vocab = topic_vocabulary(docs, labels, sample_vocabulary(docs))
    model = bertopic_model(vocab, representations)
    model.fit_transform(docs, embeddings=emb, y=labels)
    # Save-able, standard vectorizer with identical behaviour (the callable analyzer isn't JSON).
    # .fit() on a fixed vocabulary only sets vocabulary_, which BERTopic.save needs.
    model.vectorizer_model = CountVectorizer(stop_words="english", ngram_range=C.VECTORIZER_PARAMS["ngram_range"], vocabulary=vocab).fit([""])
    return model, vocab


def quality(model, docs: list) -> dict:
    words = extract_top_words_from_bertopic(model, topn=C.TOP_N_WORDS)
    _, td = topic_uniqueness_and_diversity(words, topn=C.TOP_N_WORDS)
    rng = np.random.default_rng(C.SEED)
    sample = rng.choice(len(docs), size=min(C.COHERENCE_SAMPLE, len(docs)), replace=False)
    analyzer = StreamingNgrams(C.VECTORIZER_PARAMS["ngram_range"][1])
    texts = [list(analyzer(docs[i])) for i in sample]
    # gensim needs every keyword in the corpus; a few rare n-grams are absent from the sample.
    present = {t for doc in texts for t in doc}
    kept = {k: [w for w in v if w in present] for k, v in words.items()}
    kept = {k: v for k, v in kept.items() if len(v) >= 2}
    coh = compute_cv_coherence(kept, texts, window_size=110)
    dropped = sum(len(v) for v in words.values()) - sum(len(v) for v in kept.values())
    return {"coherence_cv": round(coh["overall"], 4), "topic_diversity": round(td, 4),
            "coherence_words_dropped": int(dropped)}


def label_stats(labels: np.ndarray) -> dict:
    sizes = pd.Series(labels[labels >= 0]).value_counts()
    return {"n_topics": int(len(sizes)), "outlier_share": round(float((labels < 0).mean()), 4),
            "median_topic_size": int(sizes.median()) if len(sizes) else 0,
            "largest_topic_share": round(float(sizes.max() / len(labels)), 4) if len(sizes) else 0.0}


# ------------------------------- commands ----------------------------------------
def cmd_grid(grid: list) -> None:
    reviews, emb, u5 = load_inputs()
    docs = reviews["processed_text"].tolist()
    GRID.mkdir(parents=True, exist_ok=True)
    rows = []
    for mcs in grid:
        t0 = time.time()
        if (GRID / f"labels_mcs{mcs}.npy").exists():         # re-runs reuse the clustering
            labels, probs = np.load(GRID / f"labels_mcs{mcs}.npy"), np.load(GRID / f"probs_mcs{mcs}.npy")
        else:
            labels, probs = cluster(u5, mcs)
            np.save(GRID / f"labels_mcs{mcs}.npy", labels)
            np.save(GRID / f"probs_mcs{mcs}.npy", probs)
        row = {"min_cluster_size": mcs, **label_stats(labels), "cluster_seconds": round(time.time() - t0)}
        if row["n_topics"] >= 2:
            model, _ = fit_topics(docs, emb, labels, representations=False)
            row.update(quality(model, docs))
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(C.DATA / "step3_grid.csv", index=False)


def cmd_fit(mcs: int, reuse: bool = False) -> None:
    """Fit and save the final model; with reuse=True, load the saved model and only (re)write outputs."""
    reviews, emb, u5 = load_inputs()
    docs = reviews["processed_text"].tolist()
    t0 = time.time()
    lab_f, prob_f = GRID / f"labels_mcs{mcs}.npy", GRID / f"probs_mcs{mcs}.npy"
    if lab_f.exists():
        labels, probs = np.load(lab_f), np.load(prob_f)
    else:
        labels, probs = cluster(u5, mcs)
    out = C.DATA / "step3_bertopic_model"
    if reuse:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        model = BERTopic.load(str(out), embedding_model=SentenceTransformer(C.EMBEDDING_MODEL, device="cuda"))
        vocab = list(model.vectorizer_model.vocabulary_)
    else:
        model, vocab = fit_topics(docs, emb, labels, representations=True)
        model.save(out, serialization="safetensors", save_ctfidf=True, save_embedding_model=C.EMBEDDING_MODEL)
        model.get_topic_info().to_csv(C.DATA / "step3_topic_info.csv", index=False)
    # BERTopic renumbers topics by size (0 = largest; -1 stays outliers). Keep its numbering (it matches
    # the saved model and topic table) and the raw HDBSCAN label, after checking the map is one-to-one.
    topics = np.asarray(model.topics_)
    pairs = pd.DataFrame({"t": topics, "h": labels}).drop_duplicates()
    assert len(topics) == len(labels) and pairs.t.is_unique and pairs.h.is_unique, "topic map is not one-to-one"
    rt = reviews[["id", "store_id", "review_id"]].copy()
    rt["topic"], rt["hdbscan_label"], rt["topic_prob"] = topics, labels, probs
    path = C.DATA / "step3_review_topics.parquet"
    if path.exists():                                   # keep the frozen-arm columns
        old = pd.read_parquet(path)
        for col in ("frozen_topic", "frozen_sim"):
            if col in old.columns and (old["id"].to_numpy() == rt["id"].to_numpy()).all():
                rt[col] = old[col].to_numpy()
    rt.to_parquet(path, index=False)
    meta = {"engine": C.ENGINE, "min_cluster_size": mcs, "hdbscan": C.HDBSCAN_FIXED, **label_stats(labels),
            **quality(model, docs), "vocabulary": len(vocab), "seconds": round(time.time() - t0)}
    (C.DATA / "step3_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def cmd_frozen() -> None:
    """Teresa's saved Raleigh model applied to NC: each review gets the Raleigh topic whose topic
    embedding is most similar (what BERTopic.transform does for a safetensors-saved model, and what
    her notebook 3 used). English-only model; non-English reviews are not translated here."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    reviews, emb, _ = load_inputs()
    model = BERTopic.load(str(C.RALEIGH_MODEL), embedding_model=SentenceTransformer(C.EMBEDDING_MODEL, device="cuda"))
    te = np.asarray(model.topic_embeddings_, dtype=np.float32)
    te /= np.linalg.norm(te, axis=1, keepdims=True)
    topic_ids = np.arange(len(te)) - model._outliers
    best, sim = np.empty(len(emb), dtype=int), np.empty(len(emb), dtype=np.float32)
    for s in range(0, len(emb), 100_000):
        e = np.array(emb[s:s + 100_000], dtype=np.float32)      # copy: the memmap is read-only
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        S = e @ te.T
        best[s:s + len(e)], sim[s:s + len(e)] = topic_ids[S.argmax(axis=1)], S.max(axis=1)
    path = C.DATA / "step3_review_topics.parquet"
    rt = pd.read_parquet(path) if path.exists() else reviews[["id", "store_id", "review_id"]].copy()
    rt["frozen_topic"], rt["frozen_sim"] = best, sim
    rt.to_parquet(path, index=False)
    info = model.get_topic_info()[["Topic", "Name"]]
    counts = pd.Series(best).value_counts().rename_axis("Topic").rename("NC_count").reset_index()
    info.merge(counts, on="Topic", how="left").to_csv(C.DATA / "step3_frozen_topic_counts.csv", index=False)
    print(f"frozen Raleigh arm: {len(te)} topic embeddings; outlier share {(best == -1).mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "fit", "frozen"])
    ap.add_argument("--mcs", type=int)
    ap.add_argument("--reuse", action="store_true", help="fit: load the saved model, only write outputs")
    ap.add_argument("--grid", type=int, nargs="*", default=C.MIN_CLUSTER_GRID)
    args = ap.parse_args()
    if args.cmd == "grid":
        cmd_grid(args.grid)
    elif args.cmd == "fit":
        cmd_fit(args.mcs or C.MIN_CLUSTER_SIZE, reuse=args.reuse)
    else:
        cmd_frozen()


if __name__ == "__main__":
    main()
