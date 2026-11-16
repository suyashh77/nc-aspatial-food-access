"""Step 4, frozen arm: Teresa's saved Raleigh model and her own topic -> dimension table applied to NC reviews.

    .venv/bin/python step4_frozen.py

Each review's Raleigh topic comes from step3 (`frozen_topic`: nearest of her topic embeddings, which is what her saved
model's transform does). Its labels come from her thesis file (pred_dim, pred_dim2, pred_dim3 per topic); her 18
invalid topics are written to data/step4_frozen_invalid_topics.csv so step 6 can use her validity.
Output: data/step4_review_dims_frozen.parquet (same columns as step4_review_dims.parquet).
"""
import pandas as pd

import config as C

rt = pd.read_parquet(C.DATA / "step3_review_topics.parquet", columns=["id", "store_id", "review_id", "frozen_topic", "frozen_sim"])
th = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/raleigh_ring_reviews_with_dims.csv", low_memory=False)
tab = th[th.topic >= 0].groupby("topic")[["pred_dim", "pred_dim2", "pred_dim3"]].first()
invalid = tab.index[tab.pred_dim == "invalid topic"]
pd.DataFrame({"Topic": invalid}).to_csv(C.DATA / "step4_frozen_invalid_topics.csv", index=False)
tab = tab.drop(invalid)
out = rt.rename(columns={"frozen_topic": "topic", "frozen_sim": "topic_sim"})
out["fit_topic"] = out.topic
out = out.merge(tab, left_on="topic", right_index=True, how="left")
out[["id", "store_id", "review_id", "fit_topic", "topic", "topic_sim", "pred_dim", "pred_dim2", "pred_dim3"]].to_parquet(
    C.DATA / "step4_review_dims_frozen.parquet", index=False)
print(len(out), "reviews;", len(tab), "valid Raleigh topics;", len(invalid), "invalid")
