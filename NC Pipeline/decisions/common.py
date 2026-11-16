"""Shared loaders for the decision tests."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import pandas as pd

import config as C
from definitions import DIMS

OUT = HERE / "out"
CODE = {"AV": "Availability", "AC": "Accessibility", "AM": "Accommodation", "AF": "Affordability", "AP": "Acceptability"}


CONSENSUS = HERE / "labels" / "gold_consensus.csv"     # written by step5_import_web_labels.py


def gold(source: str = "auto") -> pd.DataFrame:
    """Gold sample with labels: dims (set of dimension names) and polarity per dimension.
    source: "auto" = human majority vote (labels/gold_consensus.csv) when it covers >= 200 reviews, else Claude's stand-in;
            "human" or "claude" to force one. The chosen source is in g.attrs["source"]."""
    g = pd.read_csv(OUT / "gold_sample.csv")
    if source == "human" or (source == "auto" and CONSENSUS.exists() and len(pd.read_csv(CONSENSUS)) >= 200):
        c = pd.read_csv(CONSENSUS, dtype={d: str for d in DIMS}).fillna("")
        lab = {int(r.gid): {d: getattr(r, d) for d in DIMS if getattr(r, d)} for r in c.itertuples()}
        g = g[g.gid.isin(lab)].copy()
        g["labels"] = g.gid.map(lab)
        g["dims"] = g["labels"].map(lambda d: set(d))
        g.attrs["source"] = f"human consensus ({len(g)} reviews)"
        return g
    rows = {}
    for f in ("gold_labels_part1.txt", "gold_labels_part2.txt"):
        for line in (OUT / f).read_text().splitlines():
            gid, *codes = line.split()
            rows[int(gid)] = {} if codes == ["none"] else {CODE[c[:2]]: c[2:] for c in codes}
    g["labels"] = g.gid.map(rows)
    g["dims"] = g["labels"].map(lambda d: set(d))
    g.attrs["source"] = "Claude stand-in"
    return g


_model = None


def st():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(C.EMBEDDING_MODEL, device="cuda")
    return _model


def review_rows(ids) -> np.ndarray:
    """Row positions of review ids in step1 (row-aligned with step1_embeddings.npy)."""
    r = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id"])
    pos = pd.Series(np.arange(len(r)), index=r["id"].to_numpy())
    return pos.loc[list(ids)].to_numpy()


def embeddings():
    return np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def score_single(pred: list, gold_sets: list) -> dict:
    """pred: one dimension per review; gold_sets: set of dims per review (non-empty).
    hit = pred is one of the gold dims. Per-dim precision/recall, macro-F1."""
    pred = list(pred)
    hit = np.mean([p in g for p, g in zip(pred, gold_sets)])
    per = {}
    for d in DIMS:
        npred = sum(p == d for p in pred)
        ngold = sum(d in g for g in gold_sets)
        tp = sum(p == d and d in g for p, g in zip(pred, gold_sets))
        prec = tp / npred if npred else 0.0
        rec = tp / ngold if ngold else 0.0
        per[d] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
                  "n_pred": npred, "n_gold": ngold}
    return {"hit_rate": round(float(hit), 3), "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 3), "per_dim": per}


SIGNAL = {1: -1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0}
LABELS = ["pred_dim", "pred_dim2", "pred_dim3"]


def invalid_topics() -> set:
    """Human check (data/step5_manual_scores.csv) when it exists; otherwise the provisional list: topics drafted as
    general praise or no clear subject."""
    f = C.DATA / "step5_manual_scores.csv"
    if f.exists():
        s = pd.read_csv(f)
        return set(range(200)) - set(s.Topic[s.manual_topic_status == "valid"])
    p = pd.read_csv(C.DATA / "parents_draft.csv")
    return set(p.topic[p.parent.str.startswith(("General praise", "No clear subject"))])


def reviews_meta() -> pd.DataFrame:
    return pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id", "store_id", "rating", "text"])


def long_dims(df: pd.DataFrame, dim_sets_col: str = "dims") -> pd.DataFrame:
    """df has store_id, rating and a column of dimension sets -> one row per (review, dim)."""
    x = df[["store_id", "rating", dim_sets_col]].explode(dim_sets_col).dropna(subset=[dim_sets_col])
    x = x.rename(columns={dim_sets_col: "dim"})
    x["signal"] = x.rating.map(SIGNAL)
    return x


def store_scores(long: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Teresa's raw store x dimension mean signal (before pooling); NaN where n < min_n."""
    g = long.groupby(["store_id", "dim"]).signal.agg(["mean", "size"]).reset_index()
    g.loc[g["size"] < min_n, "mean"] = np.nan
    return g.pivot(index="store_id", columns="dim", values="mean").reindex(columns=DIMS)


def compare_scores(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = {}
    for d in DIMS:
        j = pd.concat([a[d], b[d]], axis=1, keys=["a", "b"]).dropna()
        out[d] = {"spearman": round(float(j.a.corr(j.b, method="spearman")), 3) if len(j) > 5 else None, "stores": int(len(j))}
    return out


def score_sets(pred_sets: list, gold_sets: list) -> dict:
    """Multi-label: micro precision / recall / F1 over (review, dimension) pairs."""
    tp = sum(len(p & g) for p, g in zip(pred_sets, gold_sets))
    npred = sum(len(p) for p in pred_sets)
    ngold = sum(len(g) for g in gold_sets)
    pr, rc = (tp / npred if npred else 0), (tp / ngold if ngold else 0)
    return {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0,
            "reviews": len(gold_sets), "labels_per_review_pred": round(npred / max(len(pred_sets), 1), 2),
            "labels_per_review_gold": round(ngold / max(len(gold_sets), 1), 2)}


def dimset(row, cols=LABELS) -> set:
    return {row[c] for c in cols if isinstance(row[c], str)}
