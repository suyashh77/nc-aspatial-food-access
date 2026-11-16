"""D10. Where do the five definitions sit in the space the topics were clustered in?

Stage 4 matches topics to definitions in the 384-D embedding space; HDBSCAN found the topics in the 5-D UMAP space.
Step 2 saved the 5-D coordinates but not the fitted UMAP model, so a definition can't be transformed directly. Instead
each definition is placed at the median 5-D position of its 50 nearest reviews in 384-D (UMAP keeps nearest neighbours
close, so this is what transform would roughly do).
Question: does a definition land inside or near the topics stage 4 assigns to it, or somewhere else?

    .venv/bin/python decisions/d10_definitions_in_umap.py
Output: decisions/out/d10_definitions_in_umap.json
"""
import json

from common import *
from definitions import embed_definitions

K = 50
rng = np.random.default_rng(0)
E = embeddings()
U = np.load(C.DATA / "step2_umap5.npy", mmap_mode="r")
rd = pd.read_parquet(C.DATA / "step4_review_dims.parquet", columns=["topic"])
lab = rd.topic.to_numpy()
td = pd.read_csv(C.DATA / "step4_topic_dims.csv", index_col="Topic")
inv = invalid_topics()
D = embed_definitions(st(), C.DEFINITIONS).astype(np.float32)

# 384-D neighbours of each definition (exact, in chunks)
best_s = np.full((5, K), -np.inf, np.float32)
best_i = np.zeros((5, K), np.int64)
for s0 in range(0, len(E), 200_000):
    sims = D @ np.asarray(E[s0:s0 + 200_000]).T
    for j in range(5):
        cand_s = np.concatenate([best_s[j], sims[j]])
        cand_i = np.concatenate([best_i[j], np.arange(s0, s0 + sims.shape[1])])
        k = np.argpartition(-cand_s, K)[:K]
        best_s[j], best_i[j] = cand_s[k], cand_i[k]

# topic centres in 5-D (median of a sample of members) and their spread
cent5, spread = {}, {}
for t in np.unique(lab):
    if t < 0:
        continue
    m = np.flatnonzero(lab == t)
    m = np.sort(rng.choice(m, min(len(m), 5000), replace=False))
    u = np.asarray(U[m])
    cent5[t] = np.median(u, 0)
    spread[t] = float(np.median(np.linalg.norm(u - cent5[t], axis=1)))
tids = np.array(sorted(cent5))
C5 = np.stack([cent5[t] for t in tids])

R = {"k_neighbours": K, "definitions": {}}
for j, d in enumerate(DIMS):
    nb = np.sort(best_i[j])
    pos = np.median(np.asarray(U[nb]), 0)
    nb_topics = pd.Series(lab[nb]).value_counts()
    dist = np.linalg.norm(C5 - pos, axis=1)
    near = tids[np.argsort(dist)[:5]]
    assigned = [t for t in td.index[td.pred_dim == d] if t not in inv]
    R["definitions"][d] = {
        "mean_cosine_of_50_neighbours": round(float(best_s[j].mean()), 3),
        "neighbour_topics": {int(t): int(n) for t, n in nb_topics.head(5).items()},
        "nearest_topics_5d": [{"topic": int(t), "distance": round(float(dist[tids == t][0]), 2), "spread": round(spread[t], 2),
                               "first_dim": td.loc[t, "pred_dim"], "words": str(td.loc[t, "Name"])[:50] if "Name" in td else ""} for t in near],
        "topics_assigned_by_stage4": len(assigned),
        "share_of_neighbours_in_assigned_topics": round(float(np.isin(lab[nb], assigned).mean()), 2),
        "nearest_5d_topic_is_assigned_to_it": bool(near[0] in assigned),
        "example_neighbours": [str(x)[:140] for x in reviews_meta().text.to_numpy()[nb[:3]]],
    }
    print(d, R["definitions"][d]["neighbour_topics"], R["definitions"][d]["share_of_neighbours_in_assigned_topics"], flush=True)
(OUT / "d10_definitions_in_umap.json").write_text(json.dumps(R, indent=2, default=str))
