"""Data for the web labeler's optional third task: the human check of Pipeline B's 83 sentence topics (B2).

    .venv/bin/python apps/web_labeler/build_data_b2.py

Rebuilds the sentence list exactly as b2_sentences.py did (same split rule, same 2-word filter, same order), so row k of
b2_sentence_topics.npy is sentence k here. Shows each sentence as written (not lowercased). Per topic: keywords, size,
mean stars, the 3 sentences closest to the topic centre and 3 random ones. Draft groups: Ward clustering of
the topic centres, cut at 30 groups (Pipeline A has 25); a group whose topics are all flagged
generic by the automatic filter is named "General praise (likely invalid)", so it starts as invalid in the page.

Writes apps/web_labeler/data_b2.js and ../NC Pipeline Improved/out/b2_validation_topics.csv (Topic, draft_parent).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config as C
from step1_embed import clean_text

B = ROOT.parent / "NC Pipeline Improved" / "out"
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|(?<=[.!?])(?=[A-Z])")

rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["rating", "text"])
raw = []
for t in rv.text.to_numpy():
    for p in SPLIT.split(str(t)):
        if len(clean_text(p).strip().split()) >= 2:
            raw.append(p.strip())
stop = np.load(B / "b2_sentence_topics.npy")
assert len(raw) == len(stop), (len(raw), len(stop))
E = np.load(B / "b2_sentence_embeddings.npy", mmap_mode="r")
z = np.load(B / "b2_centroids.npz")
cid, cen = z["ids"], z["cen"]
tab = pd.read_csv(B / "b2_topics.csv", index_col=0)
rng = np.random.default_rng(11)

# draft groups
Z = linkage(cen, method="ward")                          # unit vectors, so Euclidean ranks like cosine; most even group sizes
lab = fcluster(Z, t=30, criterion="maxclust")            # about 30 groups, like the 25 for Pipeline A
groups = {}
for g in np.unique(lab):
    ts = [int(t) for t in cid[lab == g]]
    big = max(ts, key=lambda t: tab.loc[t, "n"])
    if all(tab.loc[t, "generic_auto"] for t in ts):
        name = "General praise (likely invalid)"
    else:
        name = ", ".join(str(tab.loc[big, "words"]).split(", ")[:3]).capitalize()
    for t in ts:
        groups[t] = name
# keep names unique when two clusters get the same label
seen = {}
for t in sorted(groups, key=lambda t: -tab.loc[t, "n"]):
    n = groups[t]
    if n.startswith("General praise"):
        continue
    key = (n, int(lab[list(cid).index(t)]))
    seen.setdefault(n, set()).add(key[1])
    if len(seen[n]) > 1:
        groups[t] = f"{n} ({sorted(seen[n]).index(key[1]) + 1})"

out = []
by_topic = pd.Series(np.arange(len(stop))).groupby(stop)
for t in sorted(cid, key=lambda t: -tab.loc[t, "n"]):
    members = by_topic.get_group(t).to_numpy()
    samp = np.sort(rng.choice(members, min(len(members), 20000), replace=False))
    sims = np.asarray(E[samp]) @ cen[list(cid).index(t)]
    top = [raw[i][:400] for i in samp[np.argsort(-sims)[:3]]]
    rnd = [raw[i][:400] for i in rng.choice(members, 3, replace=False)]
    out.append({"t": int(t), "n": int(tab.loc[t, "n"]), "stars": float(tab.loc[t, "mean_stars"]), "kw": str(tab.loc[t, "words"]),
                "group": groups[int(t)], "inv": bool(tab.loc[t, "generic_auto"]),
                "note": "Automatic filter: mostly generic praise words, so it starts as invalid" if tab.loc[t, "generic_auto"] else "",
                "top": top, "rand": rnd})
Path(__file__).with_name("data_b2.js").write_text("const TOPICS_B2=" + json.dumps(out, ensure_ascii=False) + ";\n")
pd.DataFrame({"Topic": [o["t"] for o in out], "draft_parent": [o["group"] for o in out], "generic_auto": [o["inv"] for o in out]}).to_csv(B / "b2_validation_topics.csv", index=False)
print(len(out), "sentence topics in", len(set(groups.values())), "draft groups")
print(pd.Series(groups).value_counts().head(12).to_string())
