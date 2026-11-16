"""Build the data block for the shareable web labeler (apps/web_labeler/labeler.html).

Reads the same inputs as the Streamlit apps and writes data.js with the reviews (in the same shuffled order the
Streamlit gold labeler uses) and the topic table. Stars, store and topic are never included for reviews; model
guesses are never included for topics.

Run: .venv/bin/python apps/web_labeler/build_data.py   then   .venv/bin/python apps/web_labeler/build_page.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
s = pd.read_csv(ROOT / "decisions/out/gold_sample.csv")[["gid", "id", "text"]]
s = s.iloc[np.random.default_rng(20261110).permutation(len(s))].reset_index(drop=True)
reviews = [{"gid": int(r.gid), "id": int(r.id), "text": str(r.text)} for r in s.itertuples()]

t = pd.read_csv(ROOT / "data/step5_validation_topics.csv").fillna("").sort_values("n_reviews", ascending=False)
topics = []
for r in t.to_dict("records"):
    topics.append({
        "t": int(r["Topic"]), "n": int(r["n_reviews"]), "stars": float(r["mean_stars"]),
        "kw": r["Representation"], "group": r["draft_parent"] or "Ungrouped", "note": r["draft_note"],
        "top": [r[f"top_review_{i}"] for i in (1, 2, 3) if str(r[f"top_review_{i}"]).strip()],
        "rand": [r[f"random_review_{i}"] for i in (1, 2, 3) if str(r[f"random_review_{i}"]).strip()],
    })
out = Path(__file__).with_name("data.js")
out.write_text("const REVIEWS=" + json.dumps(reviews, ensure_ascii=False) + ";\nconst TOPICS=" + json.dumps(topics, ensure_ascii=False) + ";\n")
print(len(reviews), "reviews,", len(topics), "topics ->", out)
