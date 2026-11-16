"""D1. Which definition text did Teresa use, and which works better?

Part A (which she used): recompute her Raleigh topic probabilities from her saved centroids with each text.
Part B (which is better): 260 hand-labelled NC reviews (gold set, labelled blind to stars and topic by Claude
using the thesis slide-9 literature definitions). Each variant is scored two ways:
  direct   - the review's own embedding vs the 5 definitions, top-1
  topic    - the review inherits its topic's top dimension (Teresa's route; Pipeline A step 4 assignment)
Part C: her own 15 hand-scored topics from the demo pipeline (9 valid): does the top-1 match her top human score?
"""
import json

from common import *
from definitions import FIRST_PERSON, THIRD_PERSON, VERSIONS, embed_definitions
import step4_match as S4

R = {}
m = st()
# ---- Part A
z = np.load(C.ROOT / "Teresa Project/Raleigh_Data/final_embedding_model/topic_centroids_frozen.npz", allow_pickle=True)
cen, ids = unit(z["centroids"]), z["topic_ids"].astype(int)
th = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/raleigh_ring_reviews_with_dims.csv", low_memory=False)
th = th[th.topic >= 0].groupby("topic")[[f"{d}_p" for d in DIMS]].first().dropna()
R["A_reproduce_thesis"] = {}
for v in ("first_person", "third_person"):
    p = S4.sims_to_probs(pd.DataFrame(cen @ embed_definitions(m, v).T, index=ids, columns=DIMS), 0.11, 0.17)
    p = p.loc[th.index, [f"{d}_p" for d in DIMS]]
    R["A_reproduce_thesis"][v] = {"topics_compared": len(th), "max_abs_prob_diff": round(float((p - th).abs().max().max()), 4),
                                  "top1_agreement": round(float((p.values.argmax(1) == th.values.argmax(1)).mean()), 3)}

# ---- Part B
g = gold()
g = g[g.dims.map(len) > 0].reset_index(drop=True)
E = unit(embeddings()[review_rows(g.id)])
rd = pd.read_parquet(C.DATA / "step4_review_dims.parquet", columns=["id", "topic"])
g = g.drop(columns=["topic"]).merge(rd, on="id", how="left")
emb = embeddings()
tcen_ids, tcen, _ = S4.centroids(emb, pd.read_parquet(C.DATA / "step4_review_dims.parquet", columns=["topic"]).topic.to_numpy())
pos = {t: i for i, t in enumerate(tcen_ids)}
in_topic = g.topic.isin(pos.keys()).to_numpy()
variants = [(v, mode) for v in VERSIONS for mode in ("paragraph", "sentences")]
R["B_gold"] = {"n_reviews_with_labels": int(len(g)), "n_random_subset": int((g.src == "random").sum()), "variants": {}}
for v, mode in variants:
    D = embed_definitions(m, v, mode).astype(np.float64)
    direct = [DIMS[i] for i in (E @ D.T).argmax(1)]
    tdim = np.array([DIMS[i] for i in (tcen @ D.T).argmax(1)])
    topic_pred = [tdim[pos[t]] for t in g.topic[in_topic]]
    key = f"{v}/{mode}"
    rand = (g.src == "random").to_numpy()
    R["B_gold"]["variants"][key] = {
        "direct_all": score_single(direct, g.dims), "direct_random_only": score_single(np.array(direct)[rand], g.dims[rand]),
        "topic_route": score_single(topic_pred, g.dims[in_topic]),
        "topic_dims_count": pd.Series(tdim).value_counts().to_dict()}
    print(key, R["B_gold"]["variants"][key]["direct_all"]["hit_rate"], R["B_gold"]["variants"][key]["direct_all"]["macro_f1"],
          "| topic", R["B_gold"]["variants"][key]["topic_route"]["hit_rate"], R["B_gold"]["variants"][key]["topic_route"]["macro_f1"])

# ---- Part C
mv = pd.read_csv(C.ROOT / "Teresa Project/Code Pipeline/data/manual_validation/manual_validation_scores.csv")
mv = mv[mv.manual_topic_status == "valid"].set_index("Topic")
dz = np.load(C.ROOT / "Teresa Project/Code Pipeline/topic_centroids_frozen.npz", allow_pickle=True)
dc, did = unit(dz["centroids"]), dz["topic_ids"].astype(int)
R["C_teresa_demo_topics"] = {"n_valid_topics": int(len(mv))}
for v in VERSIONS:
    D = embed_definitions(m, v)
    top = pd.Series([DIMS[i] for i in (dc @ D.T).argmax(1)], index=did)
    hits = [top[t] in {d for d in DIMS if mv.loc[t, f"manual_{d}"] == mv.loc[t, [f"manual_{x}" for x in DIMS]].max()} for t in mv.index]
    R["C_teresa_demo_topics"][v] = {"top1_matches_human_top": int(sum(hits))}
(OUT / "d1_definitions.json").write_text(json.dumps(R, indent=2))
print(json.dumps({k: v for k, v in R.items() if k != "B_gold"}, indent=2))
