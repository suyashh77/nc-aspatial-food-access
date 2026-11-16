"""B2. Cluster sentences instead of whole reviews (Pipeline B, experiment 2).

Problem: 45% of reviews have 2+ sentences and 79% of those touch 2+ subjects, but Pipeline A gives each review one
topic. "Good prices. Clean. Friendly service." can only count toward one subject's dimension(s).

Method:
  1. Split every review into sentences (punctuation and line breaks), clean each with Teresa's clean_text, keep
     sentences of 2+ words. Embed with the same MiniLM model.
  2. UMAP (same params) fitted on a 1M-sentence sample, applied to all; HDBSCAN min cluster 2,000 on the sample;
     every sentence then goes to its nearest topic (transform-assign, as in Pipeline A).
  3. Topic centroids -> first-person definitions (same tau, delta). Topics whose top words are >= 70% generic praise
     are set aside (automatic stand-in for the human check).
  4. A review's dimensions = union of its sentences' dimensions. Stars still give the direction.
Compared with Pipeline A: topics, gold-set multi-label precision/recall/F1, share of reviews counting toward each
dimension, store-score agreement.
"""
import re
import time

from common_b import *
from step1_embed import clean_text

t0 = time.time()
R = {}
rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id", "store_id", "rating", "text"])
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|(?<=[.!?])(?=[A-Z])")
rows, sents = [], []
for i, t in enumerate(rv.text.to_numpy()):
    for p in SPLIT.split(str(t)):
        c = clean_text(p).strip()
        if len(c.split()) >= 2:
            rows.append(i)
            sents.append(c)
rows = np.array(rows)
R["sentences"] = len(sents)
R["reviews_with_2plus_sentences"] = round(float((np.bincount(rows, minlength=len(rv)) >= 2).mean()), 3)
cache = OUT / "b2_sentence_embeddings.npy"
if cache.exists():
    E = np.load(cache)
else:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(C.EMBEDDING_MODEL, device="cuda")
    E = m.encode(sents, batch_size=2048, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(cache, E)
np.save(OUT / "b2_sentence_rows.npy", rows)
R["embed_seconds"] = round(time.time() - t0)
rng = np.random.default_rng(C.SEED)
fit = np.sort(rng.choice(len(E), min(1_000_000, len(E)), replace=False))
u5 = umap5(E, fit_rows=fit)
lab_fit = hdbscan(u5[fit], C.MIN_CLUSTER_SIZE)
R["fit"] = {"sample": len(fit), "topics": int(len(set(lab_fit)) - (1 if -1 in lab_fit else 0)), "outlier_share": round(float((lab_fit == -1).mean()), 3)}
# topic embeddings from the fitted sample, then every sentence -> nearest topic
ids = np.unique(lab_fit)
pos = {t: i for i, t in enumerate(ids)}
acc = np.zeros((len(ids), E.shape[1]))
np.add.at(acc, np.array([pos[t] for t in lab_fit]), E[fit])
te = unit(acc)
final = np.empty(len(E), dtype=int)
for s in range(0, len(E), 200_000):
    final[s:s + 200_000] = ids[(E[s:s + 200_000] @ te.T).argmax(1)]
np.save(OUT / "b2_sentence_topics.npy", final)
R["outlier_share_after_transform"] = round(float((final == -1).mean()), 4)
words = top_words(np.array(sents, dtype=object), final)
gen = generic_topics(words)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(C.EMBEDDING_MODEL, device="cuda")
D = embed_definitions(m, C.DEFINITIONS)
cid, cen = centroids(E, final)
tab = S4.sims_to_probs(pd.DataFrame(cen @ D.T, index=cid, columns=DIMS), C.SOFTMAX_TAU, C.MARGIN_DELTA)
stars = rv.rating.to_numpy()[rows]
tab["n"] = pd.Series(final).value_counts().reindex(cid).to_numpy()
tab["mean_stars"] = pd.Series(stars).groupby(final).mean().reindex(cid).round(2).to_numpy()
tab["words"] = [", ".join(words.get(int(t), [])) for t in cid]
tab["generic_auto"] = [t in gen for t in cid]
tab.to_csv(OUT / "b2_topics.csv")
R["topics_by_pred_dim"] = tab.pred_dim.value_counts().to_dict()
R["generic_topics_auto"] = len(gen)
R["stars_eta2_sentence_topics"] = stars_eta2(final, stars)
R["cashier_check_sentences"] = cashier_check(np.array(sents, dtype=object), final)
from common import dimset, gold, score_sets, store_scores, long_dims, compare_scores, invalid_topics
mp = {t: (set() if t in gen else dimset(r)) for t, r in tab.iterrows()}
sd = pd.Series([mp.get(t, set()) for t in final])
rev_dims = sd.groupby(rows).agg(lambda s: set().union(*s)).reindex(range(len(rv))).map(lambda x: x if isinstance(x, set) else set())
A = pd.read_parquet(C.DATA / "step4_review_dims.parquet")
ta = pd.read_csv(C.DATA / "step4_topic_dims.csv", index_col="Topic")
inv = invalid_topics()
ma = {t: (set() if t in inv else dimset(r)) for t, r in ta.iterrows()}
a_dims = A.topic.map(lambda t: ma.get(t, set()))
df = pd.DataFrame({"store_id": rv.store_id, "rating": rv.rating, "A": a_dims.to_numpy(), "B2": rev_dims.to_numpy()})
for c in ("A", "B2"):
    R[f"reviews_{c}"] = {"share_with_any_dim": round(float((df[c].map(len) > 0).mean()), 3),
                         "mean_dims": round(float(df[c].map(len).mean()), 2),
                         "share_by_dim": {d: round(float(df[c].map(lambda x: d in x).mean()), 3) for d in DIMS}}
R["store_scores_A_vs_B2"] = compare_scores(store_scores(long_dims(df, "A")), store_scores(long_dims(df, "B2")))
g = gold()
g = g[g.dims.map(len) > 0]
gp = pd.Series(np.arange(len(rv)), index=rv.id.to_numpy()).loc[g.id].to_numpy()
R["gold"] = {"A": score_sets(list(df.A.iloc[gp]), list(g.dims)), "B2": score_sets(list(df.B2.iloc[gp]), list(g.dims))}
R["seconds"] = round(time.time() - t0)
save("b2_sentences", R)
print(json.dumps(R, indent=1))
