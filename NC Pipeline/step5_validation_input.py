"""Step 5a: build the input table for the human topic check (Teresa notebook 4).

    .venv/bin/python step5_validation_input.py

One row per topic: size, mean stars, keywords, the 3 reviews closest to the topic centroid and 3 random
readable reviews (8-80 words), plus the draft merge group from parents_draft.csv. Model predictions are
NOT included (the check is blinded, as in Teresa's app). Topic assignment = step 4 (Teresa's transform).

Output: data/step5_validation_topics.csv
"""
import ast

import numpy as np
import pandas as pd

import config as C
import step4_match as S4

rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id", "rating", "text"])
rd = pd.read_parquet(C.DATA / "step4_review_dims.parquet", columns=["id", "topic"])
assert (rv.id.to_numpy() == rd.id.to_numpy()).all()
emb = np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")
labels = rd.topic.to_numpy()
ids, cen, n = S4.centroids(emb, labels)
info = pd.read_csv(C.DATA / "step3_topic_info.csv").set_index("Topic")
draft = pd.read_csv(C.DATA / "parents_draft.csv").set_index("topic")
nw = rv.text.str.split().str.len().to_numpy()
rng = np.random.default_rng(C.SEED)
rows = []
for t, c, k in zip(ids, cen, n):
    idx = np.flatnonzero(labels == t)
    ok = idx[(nw[idx] >= 6) & (nw[idx] <= 120)]
    e = S4.unit_chunks(emb[ok], size=len(ok) + 1).__next__()[1] if len(ok) else np.zeros((0, cen.shape[1]))
    sims = e @ c
    top, seen = [], set()
    for j in ok[np.argsort(-sims)]:
        txt = " ".join(str(rv.text.iat[j]).split())
        if txt.lower() not in seen:
            seen.add(txt.lower())
            top.append(txt)
        if len(top) == 3:
            break
    pool = idx[(nw[idx] >= 8) & (nw[idx] <= 80)]
    rand = [" ".join(str(rv.text.iat[j]).split()) for j in rng.choice(pool, min(3, len(pool)), replace=False)]
    words = ast.literal_eval(info.loc[t, "Representation"]) if t in info.index else []
    row = {"Topic": int(t), "n_reviews": int(k), "mean_stars": round(float(np.nanmean(rv.rating.to_numpy(dtype=float)[idx])), 2),
           "Name": info.loc[t, "Name"] if t in info.index else "", "Representation": ", ".join(words),
           "KeyBERT": ", ".join(ast.literal_eval(info.loc[t, "KeyBERT"])) if t in info.index else "",
           "draft_parent": draft.loc[t, "parent"] if t in draft.index else "", "draft_note": draft.loc[t, "check_note"] if t in draft.index else ""}
    for i in range(3):
        row[f"top_review_{i+1}"] = top[i] if i < len(top) else ""
        row[f"random_review_{i+1}"] = rand[i] if i < len(rand) else ""
    rows.append(row)
out = pd.DataFrame(rows).fillna("")
out.to_csv(C.DATA / "step5_validation_topics.csv", index=False)
print(len(out), "topics ->", C.DATA / "step5_validation_topics.csv")
