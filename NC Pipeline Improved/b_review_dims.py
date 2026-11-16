"""Pipeline B's review-level labels: which dimensions each review counts toward, and its text sentiment on each.

Used by b_pipeline.py (store scores) and by ../NC Pipeline/export_review_table.py (the analysis export), so both read
the same labels. A review counts toward the union of its sentences' dimensions; each sentence takes its B2 sentence
topic's dimensions under the B4 definitions; topics judged invalid (human check in out/b2_manual_scores.csv, else the
automatic generic-word filter) give no dimension.
"""
from common_b import *


def review_dims() -> tuple[pd.DataFrame, dict]:
    """Returns (rd, info). rd: one row per (review, dimension): row (= step1_reviews.parquet row), dim, text
    (mean B7 sentence sentiment, -1..+1). info: definitions, validity source and topic counts, for the run's json."""
    info = {}
    srow = np.load(OUT / "b2_sentence_rows.npy")
    stop = np.load(OUT / "b2_sentence_topics.npy")
    ssig = np.load(OUT / "b7_sentence_sentiment.npy")
    z = np.load(OUT / "b2_centroids.npz")
    cid, cen = z["ids"], z["cen"]
    dv = np.load(OUT / "b4_definition_vectors.npz")
    D, tau, delta = dv["D"], float(dv["tau"]), float(dv["delta"])
    info["definitions"] = {"variant": str(dv["name"]), "tau": tau, "delta": delta}
    tab = S4.sims_to_probs(pd.DataFrame(cen @ D.T, index=cid, columns=DIMS), tau, delta)
    man = OUT / "b2_manual_scores.csv"
    if man.exists():
        m = pd.read_csv(man)
        invalid = set(cid) - set(m.Topic[m.manual_topic_status == "valid"])
        info["validity"] = "human check of sentence topics (b2_manual_scores.csv)"
    else:
        t2 = pd.read_csv(OUT / "b2_topics.csv", index_col=0)
        invalid = set(t2.index[t2.generic_auto])
        info["validity"] = f"automatic generic-word filter ({len(invalid)} topics); human check pending"
    dims_of = {t: ([] if t in invalid else [d for d in (r.pred_dim, r.pred_dim2, r.pred_dim3) if isinstance(d, str)])
               for t, r in tab.iterrows()}
    info["topics_by_first_dim"] = tab.loc[[t for t in cid if t not in invalid], "pred_dim"].value_counts().to_dict()
    S = pd.DataFrame({"row": srow, "topic": stop, "sig": ssig})
    S["dim"] = S.topic.map(lambda t: dims_of.get(t, []))
    L = S.explode("dim").dropna(subset=["dim"])
    rd = L.groupby(["row", "dim"]).sig.mean().rename("text").reset_index()
    return rd, info
