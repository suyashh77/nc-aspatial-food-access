"""B1. Remove the sentiment direction from the embeddings, then re-cluster (Pipeline B, experiment 1).

Problem: Pipeline A's topics split by mood as well as subject (rude cashiers 1.5 stars vs friendly cashiers 4.6
stars are separate topics). The text should decide the subject; stars already give the direction.

Method:
  d = mean(unit embedding | 5 stars) - mean(unit embedding | 1 star), normalised  ("sentiment direction")
  e' = e - (e . d) d, renormalised, for every review AND every definition
  then Pipeline A's steps 2-4 unchanged: UMAP (same params) -> HDBSCAN (min cluster 2,000) -> transform-assign
  -> centroids -> first-person definitions.
Checks against Pipeline A:
  - how well a linear classifier still separates 1 vs 5 stars after the removal (how much mood is left)
  - share of star variance explained by topic (eta^2): should fall
  - rude vs friendly cashier reviews: should now share a topic
  - topics, outliers, top words; gold-set multi-label F1 of the topic route
"""
import time

from common_b import *

t0 = time.time()
R = {}
E = unit(np.load(C.DATA / "step1_embeddings.npy"))
rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id", "rating", "processed_text", "text"])
y = rv.rating.to_numpy()
d = unit(E[y == 5].mean(0) - E[y == 1].mean(0))
proj = E @ d
R["projection_vs_stars_spearman_before"] = round(float(pd.Series(proj).corr(pd.Series(y), method="spearman")), 3)

from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
i15 = np.flatnonzero((y == 1) | (y == 5))
tr = rng.choice(i15, 60_000, replace=False)
te_ = rng.choice(np.setdiff1d(i15, tr), 30_000, replace=False)


def bal_acc(X):
    clf = LogisticRegression(max_iter=300, class_weight="balanced").fit(X[tr], y[tr] == 5)
    p = clf.predict(X[te_])
    t = y[te_] == 5
    return round(float((np.mean(p[t]) + np.mean(~p[~t])) / 2), 3)


R["linear_1v5_balanced_accuracy_before"] = bal_acc(E)
E2 = unit(E - np.outer(proj, d))
R["linear_1v5_balanced_accuracy_after"] = bal_acc(E2)
R["linear_1v5_on_direction_only"] = bal_acc(proj[:, None])
del E

if (OUT / "b1_labels_fit.npy").exists():               # resume: UMAP + HDBSCAN already done
    lab = np.load(OUT / "b1_labels_fit.npy")
else:
    u5 = umap5(E2)
    R["umap_seconds"] = round(time.time() - t0)
    lab = hdbscan(u5, C.MIN_CLUSTER_SIZE)
R["fit"] = {"topics": int(len(set(lab)) - (1 if -1 in lab else 0)), "outlier_share": round(float((lab == -1).mean()), 3)}
final = transform_assign_from_labels(E2, lab)
np.save(OUT / "b1_labels_fit.npy", lab)
np.save(OUT / "b1_labels_final.npy", final)
R["outlier_share_after_transform"] = round(float((final == -1).mean()), 4)

# baseline A for the same checks
a = pd.read_parquet(C.DATA / "step4_review_dims.parquet", columns=["topic"]).topic.to_numpy()
R["stars_eta2"] = {"A": stars_eta2(a, y), "B1": stars_eta2(final, y)}
R["cashier_check"] = {"A": cashier_check(rv.text, a), "B1": cashier_check(rv.text, final)}

words = top_words(rv.processed_text.to_numpy(), final)
gen = generic_topics(words)
R["generic_topics_auto"] = sorted(gen)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(C.EMBEDDING_MODEL, device="cuda")
D = embed_definitions(m, C.DEFINITIONS)
D2 = unit(D - np.outer(D @ d, d))
cid, cen = centroids(E2, final)
tab = S4.sims_to_probs(pd.DataFrame(cen @ D2.T, index=cid, columns=DIMS), C.SOFTMAX_TAU, C.MARGIN_DELTA)
ms = pd.Series(y).groupby(final).mean()
tab["n"] = pd.Series(final).value_counts().reindex(cid).to_numpy()
tab["mean_stars"] = ms.reindex(cid).round(2).to_numpy()
tab["words"] = [", ".join(words.get(int(t), [])) for t in cid]
tab["generic_auto"] = [t in gen for t in cid]
tab.to_csv(OUT / "b1_topics.csv")
R["topics_by_pred_dim"] = tab.pred_dim.value_counts().to_dict()
R["mean_stars_spread_across_topics_sd"] = {"A": round(float(pd.Series(y).groupby(a).mean().drop(-1, errors="ignore").std()), 3),
                                           "B1": round(float(ms.drop(-1, errors="ignore").std()), 3)}
# gold
from common import gold, score_sets, dimset
g = gold()
g = g[g.dims.map(len) > 0]
pos = pd.Series(np.arange(len(rv)), index=rv.id.to_numpy()).loc[g.id].to_numpy()
mp = {t: (set() if t in gen else dimset(r)) for t, r in tab.iterrows()}
R["gold_B1"] = score_sets([mp.get(t, set()) for t in final[pos]], list(g.dims))
R["seconds"] = round(time.time() - t0)
save("b1_sentiment_axis", R)
print(json.dumps(R, indent=1))
