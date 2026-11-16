"""B3. Let the data propose the dimensions (Pipeline B, experiment 3).

Idea: clustering grouped reviews into topics nobody named in advance. Do the same one level up: group topics into
dimensions using how they behave across stores, then compare the data's dimensions with the 5 A's.
This is how psychologists found the Big Five personality traits: rate people on many adjectives, see which
adjectives rise and fall together, and let factor analysis say how many underlying traits there are.

Here: an "item" is a topic group (from parents_draft.csv), a "respondent" is a store.
  view 1 - salience:    share of a store's reviews that fall in each group (what people talk about there)
  view 2 - performance: mean star signal of the store's reviews in each group (how the store does on it)
For each view:
  1. How many dimensions? Parallel analysis: keep factors whose eigenvalue beats the same matrix with each
     column shuffled (95th percentile of 50 shuffles).
  2. What are they? Factor analysis (varimax) with that many factors; top-loading groups per factor.
  3. Do they match the 5 A's? Each factor's loading-weighted topic centroid vs the 5 definitions (cosine).
     A stable factor that matches none of them is a candidate 6th dimension.
  4. Is it stable? Refit on 50 bootstrap samples of stores; mean |cosine| between matched loading vectors.
Store category is regressed out first so a factor can't just be "dollar store vs supermarket".
"""
import sqlite3

from common_b import *
from common import invalid_topics
from sklearn.decomposition import FactorAnalysis

rng = np.random.default_rng(0)
rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["store_id", "rating"])
A = pd.read_parquet(C.DATA / "step4_review_dims.parquet", columns=["topic"])
p = pd.read_csv(C.DATA / "parents_draft.csv").set_index("topic")
inv = invalid_topics()
df = pd.DataFrame({"store": rv.store_id, "y": rv.rating.map({1: -1, 2: 0, 3: 0, 4: 0, 5: 1}), "topic": A.topic})
df = df[(df.topic >= 0) & ~df.topic.isin(inv)]
df["group"] = df.topic.map(p.parent)
con = sqlite3.connect(f"file:{C.REVIEWS_DB}?mode=ro", uri=True)
cat = pd.read_sql("select store_id, category from stores", con).set_index("store_id").category
n_store = df.groupby("store").size()
stores = n_store[n_store >= 150].index
d = df[df.store.isin(stores)]
groups = sorted(d.group.unique())

sal = pd.crosstab(d.store, d.group, normalize="index")[groups]
cnt = pd.crosstab(d.store, d.group)[groups]
perf = d.groupby(["store", "group"]).y.mean().unstack()[groups].where(cnt >= 8)
# performance: keep groups present for >= 60% of stores, then stores complete on those
keep = perf.columns[perf.notna().mean() >= 0.6]
perf = perf[keep].dropna()


def residualise(X):
    c = cat.reindex(X.index).fillna("other")
    big = c.value_counts()
    c = c.where(c.map(big) >= 30, "other")
    M = pd.get_dummies(c).astype(float).to_numpy()
    Z = (X - X.mean()) / X.std()
    beta, *_ = np.linalg.lstsq(M, Z.to_numpy(), rcond=None)
    R = Z.to_numpy() - M @ beta
    return pd.DataFrame((R - R.mean(0)) / R.std(0), index=X.index, columns=X.columns)


def parallel(Z, n=50):
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1]
    sh = []
    for _ in range(n):
        S = np.column_stack([rng.permutation(col) for col in Z.T])
        sh.append(np.sort(np.linalg.eigvalsh(np.corrcoef(S.T)))[::-1])
    thr = np.percentile(sh, 95, axis=0)
    k = int(np.argmax(ev <= thr)) if (ev <= thr).any() else len(ev)
    return k, ev[:10].round(2).tolist(), thr[:10].round(2).tolist()


def varimax(L, it=100):
    p_, k = L.shape
    Rm = np.eye(k)
    for _ in range(it):
        Lr = L @ Rm
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p_))
        Rm = u @ vt
    return L @ Rm


def fa(Z, k):
    return varimax(FactorAnalysis(k, random_state=0).fit(Z).components_.T)


# topic-group centroids, to read each factor against the 5 definitions
emb = np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")
cid, cen, cn = S4.centroids(emb, A.topic.to_numpy())
cdf = pd.DataFrame(cen, index=cid)
gcen = {}
for g, ts in p[~p.index.isin(inv)].groupby("parent").groups.items():
    ts = [t for t in ts if t in cdf.index]
    w = pd.Series(cn, index=cid).loc[ts].to_numpy(float)
    gcen[g] = unit((cdf.loc[ts].to_numpy() * w[:, None]).sum(0))
from sentence_transformers import SentenceTransformer
Dm = embed_definitions(SentenceTransformer(C.EMBEDDING_MODEL, device="cuda"), C.DEFINITIONS)

R = {"stores_used": int(len(stores)), "groups": groups}
for name, X in (("salience", sal), ("performance", perf)):
    Z = residualise(X).to_numpy()
    k, ev, thr = parallel(Z)
    k = max(k, 2)
    L = fa(Z, k)
    cols = list(X.columns)
    # bootstrap stability
    stab = []
    for _ in range(50):
        b = rng.choice(len(Z), len(Z), replace=True)
        Lb = fa(Z[b], k)
        Cm = np.abs(unit(L.T) @ unit(Lb.T).T)
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-Cm)
        stab.append(Cm[r, c])
    stab = np.mean(stab, 0)
    facs = []
    for j in range(k):
        load = pd.Series(L[:, j], index=cols)
        if load.abs().max() == -load.min():
            load = -load
        top = load.sort_values(ascending=False)
        w = top[top > 0.25]
        vec = unit(sum(w[g] * gcen[g] for g in w.index)) if len(w) else None
        sims = dict(zip(DIMS, (Dm @ vec).round(3))) if vec is not None else {}
        facs.append({"top_groups": top.head(6).round(2).to_dict(), "negative_groups": top.tail(3).round(2).to_dict(),
                     "stability": round(float(stab[j]), 2), "cosine_to_definitions": sims,
                     "best_match": max(sims, key=sims.get) if sims else None,
                     "best_cosine": max(sims.values()) if sims else None})
    R[name] = {"stores": int(len(X)), "items": len(cols), "n_factors_parallel_analysis": k,
               "eigenvalues": ev, "shuffled_95pct": thr, "factors": facs}
save("b3_discover_dimensions", R)
print(json.dumps(R, indent=1, default=float))
