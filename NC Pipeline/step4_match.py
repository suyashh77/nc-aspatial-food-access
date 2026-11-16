"""Step 4: match topics to the five access dimensions (Teresa notebook 3), Pipeline A = her method.

    .venv/bin/python step4_match.py            # uses config.DEFINITIONS
    .venv/bin/python step4_match.py --defs third_person --tag third

What it does, as in her notebook 3:
  1. Re-assign every review to a topic the way her saved model's `transform` does: the topic (including
     the outlier topic -1) whose topic embedding has the highest cosine with the review. Raleigh: 52%
     outliers at fit -> 4% after this step. Topic embeddings come from the saved step-3 model.
  2. Topic centroid = mean of its reviews' unit embeddings, renormalised (build_frozen_centroids).
  3. Cosine of each centroid with each definition embedding -> sims_to_probs (softmax tau=0.11, multi-label
     margin delta=0.17), verbatim from her cell 24.
  4. Every review inherits its topic's labels. Validity (invalid topics) comes later from the human check.

Outputs (data/): step4_review_dims[_tag].parquet, step4_topic_dims[_tag].csv, step4_meta[_tag].json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config as C
from definitions import DIMS, embed_definitions


def sims_to_probs(sim_df: pd.DataFrame, temperature: float = 0.25, margin_threshold: float | None = None,
                  top_k: int = 3) -> pd.DataFrame:
    """Teresa notebook 3 cell 24, verbatim logic."""
    X = sim_df[DIMS].to_numpy(dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    logits = Xc / max(float(temperature), 1e-12)
    logits -= logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    probs = exps / exps.sum(axis=1, keepdims=True)
    out = pd.concat([sim_df, pd.DataFrame(probs, columns=[f"{d}_p" for d in DIMS], index=sim_df.index)], axis=1)
    order = np.argsort(-probs, axis=1)
    top = np.take_along_axis(probs, order, axis=1)
    out["pred_dim"] = [DIMS[i] for i in order[:, 0]]
    out["pred_prob"] = top[:, 0]
    out["prob_margin"] = top[:, 0] - top[:, 1]
    for k in (2, 3):
        if top_k >= k:
            mask = (top[:, 0] - top[:, k - 1]) <= margin_threshold if margin_threshold is not None else np.ones(len(out), bool)
            dim = np.full(len(out), None, dtype=object)
            pr = np.full(len(out), np.nan)
            dim[mask] = [DIMS[i] for i in order[mask, k - 1]]
            pr[mask] = top[mask, k - 1]
            out[f"pred_dim{k}"], out[f"pred_prob{k}"] = dim, pr
    return out


def unit_chunks(emb, size=100_000):
    for s in range(0, len(emb), size):
        e = np.array(emb[s:s + size], dtype=np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        yield s, e


def transform_assign(emb, topic_emb: np.ndarray, topic_ids: np.ndarray):
    te = topic_emb / np.linalg.norm(topic_emb, axis=1, keepdims=True)
    best = np.empty(len(emb), dtype=int)
    sim = np.empty(len(emb), dtype=np.float32)
    for s, e in unit_chunks(emb):
        S = e @ te.T
        best[s:s + len(e)], sim[s:s + len(e)] = topic_ids[S.argmax(1)], S.max(1)
    return best, sim


def centroids(emb, labels: np.ndarray):
    ids = np.array(sorted(t for t in np.unique(labels) if t >= 0))
    pos = {t: i for i, t in enumerate(ids)}
    acc = np.zeros((len(ids), emb.shape[1]))
    n = np.zeros(len(ids), dtype=int)
    for s, e in unit_chunks(emb):
        lab = labels[s:s + len(e)]
        keep = lab >= 0
        idx = np.array([pos[t] for t in lab[keep]])
        np.add.at(acc, idx, e[keep])
        np.add.at(n, idx, 1)
    acc /= n[:, None]
    acc /= np.linalg.norm(acc, axis=1, keepdims=True)
    return ids, acc, n


def load_topic_embeddings():
    from safetensors.numpy import load_file
    d = load_file(str(C.DATA / "step3_bertopic_model" / "topic_embeddings.safetensors"))
    te = d["topic_embeddings"].astype(np.float32)
    topics = json.loads((C.DATA / "step3_bertopic_model" / "topics.json").read_text())
    outliers = 1 if -1 in set(topics["topics"]) else 0
    return te, np.arange(len(te)) - outliers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", default=C.DEFINITIONS)
    ap.add_argument("--mode", default="paragraph", choices=["paragraph", "sentences"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    emb = np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")
    rt = pd.read_parquet(C.DATA / "step3_review_topics.parquet")
    te, tids = load_topic_embeddings()
    final, fsim = transform_assign(emb, te, tids)
    ids, cen, n = centroids(emb, final)
    model = SentenceTransformer(C.EMBEDDING_MODEL, device="cuda")
    D = embed_definitions(model, args.defs, args.mode).astype(np.float64)
    sim_df = pd.DataFrame(cen @ D.T, index=ids, columns=DIMS)
    soft = sims_to_probs(sim_df, C.SOFTMAX_TAU, C.MARGIN_DELTA, 3)
    soft.insert(0, "n_reviews", n)
    soft.index.name = "Topic"
    sfx = f"_{args.tag}" if args.tag else ""
    soft.to_csv(C.DATA / f"step4_topic_dims{sfx}.csv")
    out = rt[["id", "store_id", "review_id", "topic"]].rename(columns={"topic": "fit_topic"})
    out["topic"], out["topic_sim"] = final, fsim
    out = out.merge(soft[["pred_dim", "pred_dim2", "pred_dim3"]], left_on="topic", right_index=True, how="left")
    out.to_parquet(C.DATA / f"step4_review_dims{sfx}.parquet", index=False)
    meta = {"definitions": args.defs, "mode": args.mode, "tau": C.SOFTMAX_TAU, "delta": C.MARGIN_DELTA,
            "outlier_share_fit": float((rt.topic == -1).mean()), "outlier_share_after_transform": float((final == -1).mean()),
            "topics": int(len(ids)), "pred_dim_topics": soft.pred_dim.value_counts().to_dict(),
            "pred_dim_reviews": out.pred_dim.value_counts().to_dict(), "multi_label_topics": int(soft.pred_dim2.notna().sum())}
    (C.DATA / f"step4_meta{sfx}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
