"""B7. Sentiment from the text, per dimension, instead of from the stars (Pipeline B, switch 7).

    ../NC\\ Pipeline/.venv/bin/python b7_text_sentiment.py

Problem: Teresa's signal comes from the stars (1 star = -1, 2-4 = 0, 5 = +1), one value for the whole review. "Good
selection but prices are high" (3 stars) scores 0 on Availability and 0 on Affordability, and a 5-star review that
complains about parking scores +1 on Accessibility.

Method (no new model download; uses the MiniLM vectors already computed):
  1. Train a 3-class logistic regression (low = 1 star, neutral = 2-4, high = 5) on 400K review vectors; check on 100K
     held-out reviews. A text's sentiment = P(high) - P(low), on the same -1..+1 scale as the star signal.
  2. Apply it to every sentence vector from B2 (2.1M sentences).
  3. A review's sentiment on a dimension = mean sentiment of its sentences whose B2 topic maps to that dimension.
Checks: polarity on the gold set (does text or stars get "+" vs "-" right, especially for 2-4 star reviews),
how many reviews are mixed (one dimension clearly positive, another clearly negative), store-score agreement with stars.

Outputs: out/b7_review_dim_sentiment.parquet (row = step1 row, dim, text_signal, n_sentences), out/b7_text_sentiment.json
"""
import time

from sklearn.linear_model import LogisticRegression

from common_b import *
from common import gold, invalid_topics, review_rows

t0 = time.time()
R = {}
rng = np.random.default_rng(3)
E = np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")
rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id", "store_id", "rating"])
star = rv.rating.to_numpy()
ok = np.flatnonzero(~np.isnan(star))
idx = rng.permutation(ok)
tr, te = np.sort(idx[:400_000]), np.sort(idx[400_000:500_000])
cls = lambda s: np.where(s == 1, 0, np.where(s == 5, 2, 1))
clf = LogisticRegression(max_iter=300, C=1.0)
clf.fit(np.asarray(E[tr]), cls(star[tr]))
P = clf.predict_proba(np.asarray(E[te]))
pred_sig = P[:, 2] - P[:, 0]
true_sig = pd.Series(star[te]).map({1: -1, 2: 0, 3: 0, 4: 0, 5: 1}).to_numpy()
R["probe"] = {"train": len(tr), "test": len(te), "accuracy_3class": round(float((P.argmax(1) == cls(star[te])).mean()), 3),
              "majority_baseline": round(float(np.bincount(cls(star[te])).max() / len(te)), 3),
              "pearson_text_vs_star_signal": round(float(np.corrcoef(pred_sig, true_sig)[0, 1]), 3),
              "pearson_text_vs_stars_1to5": round(float(np.corrcoef(pred_sig, star[te])[0, 1]), 3)}
print(R["probe"], flush=True)

# sentences -> sentiment
Es = np.load(OUT / "b2_sentence_embeddings.npy", mmap_mode="r")
srow = np.load(OUT / "b2_sentence_rows.npy")
stop = np.load(OUT / "b2_sentence_topics.npy")
ssig = np.empty(len(Es), np.float32)
for s in range(0, len(Es), 250_000):
    p = clf.predict_proba(np.asarray(Es[s:s + 250_000]))
    ssig[s:s + 250_000] = p[:, 2] - p[:, 0]
np.save(OUT / "b7_sentence_sentiment.npy", ssig)                  # reused by B3 (performance view)
tab = pd.read_csv(OUT / "b2_topics.csv", index_col=0)
dims_of = {t: ([] if r.generic_auto else [d for d in (r.pred_dim, r.pred_dim2, r.pred_dim3) if isinstance(d, str)]) for t, r in tab.iterrows()}
S = pd.DataFrame({"row": srow, "topic": stop, "sig": ssig})
S["dims"] = S.topic.map(lambda t: dims_of.get(t, []))
L = S.explode("dims").dropna(subset=["dims"]).rename(columns={"dims": "dim"})
rd = L.groupby(["row", "dim"]).sig.agg(text_signal="mean", n_sentences="size").reset_index()
rd["text_signal"] = rd.text_signal.astype(np.float32)
rd.to_parquet(OUT / "b7_review_dim_sentiment.parquet", index=False)
whole = pd.Series(ssig).groupby(srow).mean()                       # whole-review text sentiment (fallback)

# mixed reviews
w = rd.pivot(index="row", columns="dim", values="text_signal")
mixed = ((w > 0.3).any(axis=1) & (w < -0.3).any(axis=1))
R["reviews_with_dim_sentiment"] = int(w.shape[0])
R["mixed_reviews_share"] = round(float(mixed.mean()), 3)
mid = pd.Series(star).reindex(w.index).between(2, 4)
R["mixed_share_among_2to4_star"] = round(float(mixed[mid.to_numpy()].mean()), 3)

# gold polarity
g = gold()
g["row"] = review_rows(g.id)
pairs = [(r.row, d, p, r.rating) for r in g.itertuples() for d, p in r.labels.items() if p in ("+", "-")]
key = rd.set_index(["row", "dim"]).text_signal
res = {"pairs": len(pairs)}
for name, sub in (("all", pairs), ("2-4 stars", [x for x in pairs if 2 <= x[3] <= 4])):
    y = np.array([1 if p == "+" else -1 for _, _, p, _ in sub])
    txt = np.array([key.get((r, d), whole.get(r, 0.0)) for r, d, _, _ in sub])
    st = np.array([{1: -1, 5: 1}.get(int(s), 0) if s == s else 0 for *_, s in sub])
    res[name] = {"pairs": len(sub), "text_correct": round(float((np.sign(txt) == y).mean()), 3),
                 "stars_correct": round(float((st == y).mean()), 3),
                 "stars_neutral_share": round(float((st == 0).mean()), 3),
                 "text_from_dim_sentences_share": round(float(np.mean([(r, d) in key.index for r, d, _, _ in sub])), 3)}
R["gold_polarity"] = res
R["gold_source"] = g.attrs.get("source")
print(res, flush=True)

# store scores: text vs stars on the same B2 dimension sets
rd["store_id"] = rv.store_id.to_numpy()[rd.row]
rd["star_signal"] = pd.Series(star[rd.row]).map({1: -1, 2: 0, 3: 0, 4: 0, 5: 1}).to_numpy()
cmp_ = {}
for d in DIMS:
    x = rd[rd.dim == d].groupby("store_id").agg(text=("text_signal", "mean"), stars=("star_signal", "mean"), n=("row", "size"))
    x = x[x.n >= 20]
    cmp_[d] = {"stores": int(len(x)), "spearman_text_vs_stars": round(float(x.text.corr(x.stars, method="spearman")), 3)}
R["store_scores_text_vs_stars"] = cmp_
R["seconds"] = round(time.time() - t0)
save("b7_text_sentiment", R)
print(json.dumps(R, indent=1))
