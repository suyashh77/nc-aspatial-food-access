"""B3 rerun: how many dimensions do the reviews contain, using real topics instead of hand-made groups.

    ../NC\\ Pipeline/.venv/bin/python b3_topics.py --base b1     # B1 topics (mood removed), one topic per review
    ../NC\\ Pipeline/.venv/bin/python b3_topics.py --base b2     # B2 sentence topics; performance from B7 text sentiment

Same method as b3_discover_dimensions.py (section 7 of PIPELINE_GUIDE.md): items = topics (generic ones dropped),
respondents = stores with >= 150 reviews, category regressed out, parallel analysis for the number of factors,
varimax factor analysis, 50 bootstrap refits for stability, each factor read against the 5 definitions.
New here: each factor is named from words, not just numbers: its top-loading topics' keywords and the 4 reviews
closest to the factor's direction in embedding space (section 8, "from vectors back to words").
Output: out/b3_topics_<base>.json
"""
import argparse
import sqlite3

from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis

from common_b import *

ap = argparse.ArgumentParser()
ap.add_argument("--base", choices=["b1", "b2"], required=True)
args = ap.parse_args()
rng = np.random.default_rng(0)
rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["store_id", "rating", "text"])
sig = rv.rating.map({1: -1, 2: 0, 3: 0, 4: 0, 5: 1}).to_numpy()
emb = np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")

if args.base == "b1":
    lab = np.load(OUT / "b1_labels_final.npy")
    tab = pd.read_csv(OUT / "b1_topics.csv", index_col=0)
    df = pd.DataFrame({"store": rv.store_id.to_numpy(), "y": sig, "topic": lab, "row": np.arange(len(rv))})
else:
    lab = np.load(OUT / "b2_sentence_topics.npy")
    srow = np.load(OUT / "b2_sentence_rows.npy")
    tab = pd.read_csv(OUT / "b2_topics.csv", index_col=0)
    df = pd.DataFrame({"store": rv.store_id.to_numpy()[srow], "y": np.load(OUT / "b7_sentence_sentiment.npy"), "topic": lab, "row": srow})
gen = set(tab.index[tab.generic_auto])
df = df[(df.topic >= 0) & ~df.topic.isin(gen)]
n_rev = rv.groupby("store_id").size()
stores = n_rev[n_rev >= 150].index
d = df[df.store.isin(stores)]
items = sorted(d.topic.unique())
con = sqlite3.connect(f"file:{C.REVIEWS_DB}?mode=ro", uri=True)
cat = pd.read_sql("select store_id, category from stores", con).set_index("store_id").category

sal = pd.crosstab(d.store, d.topic, normalize="index")[items]
cnt = pd.crosstab(d.store, d.topic)[items]
perf = d.groupby(["store", "topic"]).y.mean().unstack()[items].where(cnt >= 8)
keep = perf.columns[perf.notna().mean() >= 0.6]
perf = perf[keep].dropna()


def residualise(X):
    c = cat.reindex(X.index).fillna("other")
    c = c.where(c.map(c.value_counts()) >= 30, "other")
    M = pd.get_dummies(c).astype(float).to_numpy()
    Z = (X - X.mean()) / X.std()
    beta, *_ = np.linalg.lstsq(M, Z.to_numpy(), rcond=None)
    Rm = Z.to_numpy() - M @ beta
    return (Rm - Rm.mean(0)) / Rm.std(0)


def parallel(Z, n=50):
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1]
    sh = [np.sort(np.linalg.eigvalsh(np.corrcoef(np.column_stack([rng.permutation(c) for c in Z.T]).T)))[::-1] for _ in range(n)]
    thr = np.percentile(sh, 95, axis=0)
    return (int(np.argmax(ev <= thr)) if (ev <= thr).any() else len(ev)), ev[:12].round(2).tolist(), thr[:12].round(2).tolist()


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


# topic centroids in the raw review space (for reading factors against definitions and finding nearest reviews)
cent = {}
for t, g in d.groupby("topic").row:
    r = np.sort(rng.choice(g.to_numpy(), min(len(g), 3000), replace=False))
    cent[t] = unit(np.asarray(emb[r]).mean(0))
from sentence_transformers import SentenceTransformer
from definitions import embed_definitions
Dm = embed_definitions(SentenceTransformer(C.EMBEDDING_MODEL, device="cuda"), C.DEFINITIONS)
samp = np.sort(rng.choice(len(rv), 200_000, replace=False))
Es = np.asarray(emb[samp])
texts = rv.text.to_numpy()

R = {"base": args.base, "stores": int(len(stores)), "items_salience": len(items), "items_performance": int(perf.shape[1])}
for name, X in (("salience", sal), ("performance", perf)):
    if X.shape[1] < 3:
        R[name] = {"skipped": f"only {X.shape[1]} topics present at 60% of stores"}
        continue
    Z = residualise(X)
    k, ev, thr = parallel(Z)
    k = max(k, 2)
    L = fa(Z, k)
    stab = []
    for _ in range(50):
        b = rng.choice(len(Z), len(Z), replace=True)
        Cm = np.abs(unit(L.T) @ unit(fa(Z[b], k).T).T)
        r, c = linear_sum_assignment(-Cm)
        stab.append(Cm[r, c])
    stab = np.mean(stab, 0)
    cols = list(X.columns)
    facs = []
    for j in range(k):
        load = pd.Series(L[:, j], index=cols)
        if load.abs().max() == -load.min():
            load = -load
        top = load.sort_values(ascending=False)
        w = top[top > 0.25]
        vec = unit(sum(w[t] * cent[t] for t in w.index)) if len(w) else None
        sims = dict(zip(DIMS, (Dm @ vec).round(3))) if vec is not None else {}
        near = [texts[samp[i]][:160] for i in np.argsort(-(Es @ vec))[:4]] if vec is not None else []
        facs.append({"stability": round(float(stab[j]), 2),
                     "top_topics": {int(t): {"loading": round(float(v), 2), "words": str(tab.loc[t, "words"])[:90], "n": int(tab.loc[t, "n"])} for t, v in top.head(6).items()},
                     "negative_topics": {int(t): str(tab.loc[t, "words"])[:60] for t in top.tail(2).index},
                     "cosine_to_definitions": sims, "best_match": max(sims, key=sims.get) if sims else None,
                     "best_cosine": max(sims.values()) if sims else None, "nearest_reviews": near})
    R[name] = {"stores": int(len(X)), "items": len(cols), "n_factors_parallel_analysis": k, "eigenvalues": ev, "shuffled_95pct": thr, "factors": facs}
    print(name, k, [f["best_match"] for f in facs], [f["stability"] for f in facs], flush=True)
save(f"b3_topics_{args.base}", R)
