"""D8. Raleigh reproduction through stage 6 (pooled store scores), both arms, against her thesis store_scores.csv."""
import json
from common import *

her = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/store_scores.csv")
th = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/raleigh_ring_reviews_with_dims.csv", low_memory=False, usecols=["placeId", "fid"]).drop_duplicates("placeId")
her = her.merge(th, on="placeId", how="left")
# her _z: z of raw means or of pooled means?
R = {"her_z_is": {d: {"corr_with_raw_mean": round(float(her[f"{d}_z"].corr(her[f"{d}_mean"])), 3),
                       "corr_with_pooled_mean": round(float(her[f"{d}_z"].corr(her[f"{d}_pooled_mean"])), 3)} for d in DIMS}}
for arm, tag in (("frozen", "raleigh_frozen"), ("nc", "raleigh")):
    ours = pd.read_csv(C.DATA / f"step6_store_scores_{tag}.csv")
    m = pd.concat([her.merge(ours, left_on="placeId", right_on="place_id"),
                   her.merge(ours, left_on="fid", right_on="feature_id")]).drop_duplicates("placeId")
    R[arm] = {"stores": len(m)}
    for d in DIMS:
        j = m[[f"{d}_pooled_mean", f"{d}_pooled_mean_signal", f"{d}_mean", f"{d}_raw_mean_signal", f"{d}_n", f"{d}_raw_mean_signal"]].dropna()
        R[arm][d] = {"spearman_pooled": round(float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")), 3),
                     "spearman_raw": round(float(j.iloc[:, 2].corr(j.iloc[:, 3], method="spearman")), 3),
                     "her_mean_pooled": round(float(j.iloc[:, 0].mean()), 3), "our_mean_pooled": round(float(j.iloc[:, 1].mean()), 3),
                     "stores": len(j)}
(OUT / "d8_raleigh_reproduction.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
