"""Decision tests D2-D7. Each writes out/<name>.json.

    ../.venv/bin/python d2_d7.py sentences|topics|outliers|merge|arms|language|all
"""
import json
import re
import sys

from common import *
import step4_match as S4
from definitions import embed_definitions
from step1_embed import clean_text

R4 = C.DATA / "step4_review_dims.parquet"


def table_a():
    t = pd.read_csv(C.DATA / "step4_topic_dims.csv", index_col="Topic")
    return t


def route_sets(topics: np.ndarray, table: pd.DataFrame, invalid: set) -> list:
    m = {t: (set() if t in invalid else dimset(r)) for t, r in table.iterrows()}
    return [m.get(t, set()) for t in topics]


def base_frame():
    rv = reviews_meta()
    rd = pd.read_parquet(R4)
    assert (rv.id.to_numpy() == rd.id.to_numpy()).all()
    rd["rating"], rd["text"] = rv.rating.to_numpy(), rv.text.to_numpy()
    return rd


def save(name, R):
    (OUT / f"{name}.json").write_text(json.dumps(R, indent=2, default=str))
    print(name, json.dumps(R, indent=1, default=str)[:3000])


# --------------------------------------------------------------- D2 sentences (the Nov 4 memo)
def split_sentences(t: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?<=[.!?])(?=[A-Z])", str(t))
    return [p for p in (clean_text(x).strip() for x in parts) if len(p.split()) >= 2]


def d_sentences():
    rd = base_frame()
    inv = invalid_topics()
    tab = table_a()
    te, tids = S4.load_topic_embeddings()
    te = unit(te).astype(np.float32)
    rng = np.random.default_rng(1)
    counts = rd.store_id.value_counts()
    stores = rng.choice(counts[counts >= 100].index, 400, replace=False)
    g = gold()
    g = g[g.dims.map(len) > 0]
    sub = rd[rd.store_id.isin(stores) | rd.id.isin(g.id)].copy()
    sents = sub.text.map(split_sentences)
    flat = [(i, s) for i, ss in zip(sub.index, sents) for s in ss]
    idx = np.array([i for i, _ in flat])
    E = st().encode([s for _, s in flat], batch_size=1024, normalize_embeddings=True, show_progress_bar=False)
    S = E @ te.T
    st_topic = tids[S.argmax(1)]
    m = {t: (set() if t in inv else dimset(r)) for t, r in tab.iterrows()}
    sent_dims = pd.Series([m.get(t, set()) for t in st_topic], index=idx)
    union = sent_dims.groupby(level=0).agg(lambda s: set().union(*s))
    sub["dims_sent"] = union.reindex(sub.index).map(lambda x: x if isinstance(x, set) else set())
    sub["dims_review"] = route_sets(sub.topic.to_numpy(), tab, inv)
    sub["n_sent"] = sents.map(len)
    R = {"stores": len(stores), "reviews": int(sub.store_id.isin(stores).sum()), "sentences": len(flat)}
    s4 = sub[sub.store_id.isin(stores)]
    for col in ("dims_review", "dims_sent"):
        R[col] = {"share_reviews_with_any_dim": round(float((s4[col].map(len) > 0).mean()), 3),
                  "mean_dims_per_review": round(float(s4[col].map(len).mean()), 2),
                  "share_reviews_by_dim": {d: round(float(s4[col].map(lambda x: d in x).mean()), 3) for d in DIMS}}
    a = store_scores(long_dims(s4, "dims_review"))
    b = store_scores(long_dims(s4, "dims_sent"))
    R["store_score_agreement_review_vs_sentence"] = compare_scores(a, b)
    gg = sub.merge(g[["id", "dims"]], on="id")
    R["gold"] = {"review_route": score_sets(list(gg.dims_review), list(gg.dims)),
                 "sentence_route": score_sets(list(gg.dims_sent), list(gg.dims))}
    save("d2_sentences", R)


# --------------------------------------------------------------- D3 number of topics
def d_topics():
    emb = embeddings()
    rv = reviews_meta()
    g = gold()
    g = g[g.dims.map(len) > 0]
    gpos = review_rows(g.id)
    m = st()
    D = embed_definitions(m, C.DEFINITIONS).astype(np.float64)
    R, scores = {}, {}
    for mcs in (1000, 2000, 4000):
        lab = np.load(C.DATA / "step3_grid" / f"labels_mcs{mcs}.npy")
        ids_all = np.unique(lab)
        # BERTopic topic embeddings = mean member embedding (incl. the outlier topic), then transform-assign
        acc = np.zeros((len(ids_all), emb.shape[1]))
        n = np.zeros(len(ids_all))
        pos = {t: i for i, t in enumerate(ids_all)}
        for s, e in S4.unit_chunks(emb):
            li = np.array([pos[t] for t in lab[s:s + len(e)]])
            np.add.at(acc, li, e)
            np.add.at(n, li, 1)
        te = (acc / n[:, None]).astype(np.float32)
        final, _ = S4.transform_assign(emb, te, ids_all)
        cid, cen, cn = S4.centroids(emb, final)
        tab = S4.sims_to_probs(pd.DataFrame(cen @ D.T, index=cid, columns=DIMS), C.SOFTMAX_TAU, C.MARGIN_DELTA)
        sets = route_sets(final, tab, set())
        df = pd.DataFrame({"store_id": rv.store_id, "rating": rv.rating, "dims": sets})
        scores[mcs] = store_scores(long_dims(df))
        top1 = pd.Series(tab.pred_dim).reindex(final[gpos])
        ok = top1.notna().to_numpy()
        R[mcs] = {"topics": int(len(cid)), "outlier_share_after_transform": round(float((final == -1).mean()), 4),
                  "topics_by_pred_dim": tab.pred_dim.value_counts().to_dict(),
                  "gold_top1": score_single(list(top1[ok]), list(g.dims[ok])),
                  "gold_multilabel": score_sets([sets[i] for i in gpos], list(g.dims))}
        R[mcs]["gold_top1"].pop("per_dim")
    R["store_scores_vs_2000"] = {k: compare_scores(scores[2000], scores[k]) for k in (1000, 4000)}
    save("d3_topics", R)


# --------------------------------------------------------------- D4 outliers
def d_outliers():
    rd = base_frame()
    inv = invalid_topics()
    tab = table_a()
    rd["dims_all"] = route_sets(rd.topic.to_numpy(), tab, inv)
    rd["dims_fit"] = route_sets(np.where(rd.fit_topic == -1, -1, rd.topic), tab, inv)
    rd["was_outlier"] = rd.fit_topic == -1
    R = {"fit_outlier_share": round(float(rd.was_outlier.mean()), 3),
         "share_moved_to_a_different_topic_than_fit": round(float(((rd.fit_topic != rd.topic) & ~rd.was_outlier).mean()), 3)}
    for c in ("dims_fit", "dims_all"):
        R[c] = {"reviews_with_dim": int((rd[c].map(len) > 0).sum()),
                "stores_with_20plus_by_dim": store_scores(long_dims(rd, c)).notna().sum().to_dict()}
    R["store_score_agreement_fit_vs_all"] = compare_scores(store_scores(long_dims(rd, "dims_fit")), store_scores(long_dims(rd, "dims_all")))
    R["topic_sim_median"] = {"members": float(rd.topic_sim[~rd.was_outlier].median()), "reassigned_outliers": float(rd.topic_sim[rd.was_outlier].median())}
    g = gold()
    g = g[g.dims.map(len) > 0].merge(rd[["id", "dims_all", "was_outlier"]], on="id")
    for w, name in ((False, "gold_members"), (True, "gold_reassigned_outliers")):
        x = g[g.was_outlier == w]
        R[name] = score_sets(list(x.dims_all), list(x.dims))
    save("d4_outliers", R)


# --------------------------------------------------------------- D5 merge near-duplicates
def d_merge():
    rd = base_frame()
    tab = table_a()
    inv = invalid_topics()
    p = pd.read_csv(C.DATA / "parents_draft.csv").set_index("topic")
    emb = embeddings()
    cid, cen, _ = S4.centroids(emb, rd.topic.to_numpy())
    cdf = pd.DataFrame(cen, index=cid)
    R = {"parents": {}}
    valid = [t for t in tab.index if t not in inv]
    same = []
    for par, grp in p.loc[valid].groupby("parent"):
        ts = list(grp.index)
        dims = tab.loc[ts, "pred_dim"]
        same.append(dims.nunique() == 1)
        R["parents"][par] = {"topics": ts, "pred_dims": dims.to_dict()}
    R["parent_groups_all_same_top_dim"] = f"{sum(same)} of {len(same)}"
    S = cen @ cen.T
    np.fill_diagonal(S, -1)
    pairs = [(cid[i], cid[j], float(S[i, j])) for i in range(len(cid)) for j in range(i + 1, len(cid)) if S[i, j] >= 0.85]
    R["pairs_cos_ge_0.85"] = len(pairs)
    R["pairs_same_top_dim"] = sum(tab.loc[a, "pred_dim"] == tab.loc[b, "pred_dim"] for a, b, _ in pairs)
    # merged model: each valid parent becomes one topic; re-match its centroid; compare store scores
    m = st()
    D = embed_definitions(m, C.DEFINITIONS).astype(np.float64)
    par_of = {t: p.loc[t, "parent"] for t in valid}
    merged_dims = {}
    for par in set(par_of.values()):
        ts = [t for t in valid if par_of[t] == par]
        w = tab.loc[ts, "n_reviews"].to_numpy(float)
        c = unit((cdf.loc[ts].to_numpy() * w[:, None]).sum(0))
        r = S4.sims_to_probs(pd.DataFrame([c @ D.T], columns=DIMS), C.SOFTMAX_TAU, C.MARGIN_DELTA).iloc[0]
        for t in ts:
            merged_dims[t] = dimset(r)
    rd["dims_a"] = route_sets(rd.topic.to_numpy(), tab, inv)
    rd["dims_m"] = [merged_dims.get(t, set()) for t in rd.topic]
    R["store_score_agreement_unmerged_vs_merged"] = compare_scores(store_scores(long_dims(rd, "dims_a")), store_scores(long_dims(rd, "dims_m")))
    R["topics_to_review_unmerged_vs_merged"] = [len(tab), len(set(par_of.values())) + len(inv)]
    save("d5_merge", R)


# --------------------------------------------------------------- D6 frozen Raleigh arm vs NC arm; Raleigh reproduction
def d_arms():
    import sqlite3
    rd = base_frame()
    rt = pd.read_parquet(C.DATA / "step3_review_topics.parquet", columns=["id", "frozen_topic"])
    rd["frozen_topic"] = rt.frozen_topic.to_numpy()
    th = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/raleigh_ring_reviews_with_dims.csv", low_memory=False)
    ft = th[th.topic >= 0].groupby("topic")[LABELS].first()
    finv = set(ft.index[ft.pred_dim == "invalid topic"])
    rd["dims_frozen"] = route_sets(rd.frozen_topic.to_numpy(), ft, finv)
    rd["dims_nc"] = route_sets(rd.topic.to_numpy(), table_a(), invalid_topics())
    con = sqlite3.connect(f"file:{C.REVIEWS_DB}?mode=ro", uri=True)
    stores = pd.read_sql("select store_id, place_id, feature_id, category from stores", con)
    rd = rd.merge(stores[["store_id", "category"]], on="store_id", how="left")
    R = {}
    for c in ("dims_frozen", "dims_nc"):
        R[c] = {"share_reviews_with_dim": round(float((rd[c].map(len) > 0).mean()), 3),
                "share_by_dim": {d: round(float(rd[c].map(lambda x: d in x).mean()), 3) for d in DIMS}}
    top = rd.category.value_counts().index[:8]
    R["share_with_dim_by_category"] = {k: {"frozen": round(float((v.dims_frozen.map(len) > 0).mean()), 3),
                                           "nc": round(float((v.dims_nc.map(len) > 0).mean()), 3), "reviews": len(v)}
                                       for k, v in rd[rd.category.isin(top)].groupby("category")}
    g = gold()
    g = g[g.dims.map(len) > 0].merge(rd[["id", "dims_frozen", "dims_nc"]], on="id")
    R["gold"] = {c: score_sets(list(g[c]), list(g.dims)) for c in ("dims_frozen", "dims_nc")}
    R["store_scores_frozen_vs_nc"] = compare_scores(store_scores(long_dims(rd, "dims_frozen")), store_scores(long_dims(rd, "dims_nc")))
    # Raleigh reproduction: her 49 stores, matched by place_id
    ss = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/store_scores.csv")
    # match on Google place_id, else on feature_id (her `fid`): place IDs change over time, feature IDs less so
    fid = th.groupby("placeId").fid.first()
    ss["fid"] = ss.placeId.map(fid)
    by_pid = ss.merge(stores, left_on="placeId", right_on="place_id")
    by_fid = ss[~ss.placeId.isin(by_pid.placeId)].merge(stores, left_on="fid", right_on="feature_id")
    mt = pd.concat([by_pid, by_fid]).drop_duplicates("placeId")
    sub = rd[rd.store_id.isin(mt.store_id)]
    her = mt.set_index("store_id")[[f"{d}_pooled_mean" for d in DIMS]]
    her.columns = DIMS
    R["raleigh"] = {"her_stores": len(ss), "matched_in_nc": int(len(mt)), "nc_reviews_for_them": int(len(sub))}
    for c in ("dims_frozen", "dims_nc"):
        R["raleigh"][c] = compare_scores(her, store_scores(long_dims(sub, c), min_n=5))
    save("d6_arms", R)


# --------------------------------------------------------------- D7 non-English reviews
def d_language():
    import langid
    rv = reviews_meta()
    s = rv.sample(60_000, random_state=2)
    s["nw"] = s.text.str.split().str.len()
    s["lang"] = [langid.classify(t)[0] for t in s.text]
    R = {"sample": len(s)}
    for k, x in (("all", s), ("8plus_words", s[s.nw >= 8])):
        vc = x.lang.value_counts(normalize=True)
        R[k] = {"n": len(x), "english_share": round(float(vc.get("en", 0)), 4), "top_non_english": vc.drop("en", errors="ignore").head(6).round(4).to_dict()}
    es = s[(s.lang == "es") & (s.nw >= 8)].text.head(5).tolist()
    R["spanish_examples"] = es
    save("d7_language", R)


if __name__ == "__main__":
    cmds = {"sentences": d_sentences, "topics": d_topics, "outliers": d_outliers, "merge": d_merge, "arms": d_arms, "language": d_language}
    for c in (cmds if sys.argv[1] == "all" else sys.argv[1:]):
        cmds[c]()
