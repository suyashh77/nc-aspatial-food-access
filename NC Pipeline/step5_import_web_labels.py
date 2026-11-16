"""Import the hand labels from the shareable web labeler (apps/web_labeler, https://claude.ai/artifact/7cCD2HedpJ97FgEmnLo2uH).

    .venv/bin/python step5_import_web_labels.py [--src data/web_labels]

Input: the page's database, dumped as one JSON file per rater (Claude pulls it with ArtifactData list ... out_dir=data/web_labels):
    data/web_labels/gold/<rater id>.json     {"data": {"code": "SS", "labels": {"<gid>": {"id":..., "Availability": "+", ..., "none": false, ...}}}}
    data/web_labels/topics/<rater id>.json   {"data": {"code": "SS", "topics": {"<t>": {"group","status","reason"}}, "groups": {"<key>": {"name","members","status","reason","scores","notes","at"}}}}
(A bare body without the "data" wrapper is accepted too.)

Outputs:
    decisions/labels/<code>.csv            one per rater, the Streamlit labeler's columns (gid, id, rater, no_access_content, <Dim>, ...)
    decisions/labels/gold_consensus.csv    majority vote over raters (>= 3 raters per review): a dimension counts when more than
                                           half the raters mark it; its polarity is the most common mark among them.
                                           decisions/common.gold() uses this file in place of Claude's stand-in labels.
    data/step5_manual_scores_<code>.csv    one per topic rater, Teresa's per-topic format (each topic takes its group's scores)
    data/step5_manual_scores.csv           consensus over topic raters, read by stage 6: a topic is invalid when more than half
                                           its raters say so; scores are the median over raters who kept it (rounded half up).
    ../NC Pipeline Improved/out/b2_manual_scores[_<code>].csv   the same for Pipeline B's sentence topics (data/web_labels/topics_b2/),
                                           read by b_pipeline.py in place of the automatic generic filter.
    data/step5_import_meta.json            who labeled what, counts, and agreement summaries.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from definitions import DIMS

LABELS = C.HERE / "decisions" / "labels"
B_OUT = C.ROOT / "NC Pipeline Improved" / "out"
GENERIC = ("General praise", "No clear subject")
MIN_CONSENSUS_RATERS = 3


def body(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d.get("data", d) if isinstance(d, dict) else {}


def safe(code: str, rid: str) -> str:
    c = re.sub(r"[^A-Za-z0-9_-]+", "", code or "")
    return c or "rater_" + rid[-6:]


# ------------------------------------------------------------------ gold set
def gold_rows(rid: str, doc: dict) -> pd.DataFrame:
    rows = []
    for gid, l in (doc.get("labels") or {}).items():
        rows.append({"gid": int(gid), "id": l.get("id"), "rater": doc.get("code", ""), "rater_id": rid,
                     "no_access_content": bool(l.get("none")), **{d: l.get(d, "") or "" for d in DIMS},
                     "other_subject": l.get("other", ""), "confidence": l.get("conf", ""), "notes": l.get("notes", ""),
                     "seconds": l.get("sec"), "saved_at": l.get("at", "")})
    return pd.DataFrame(rows)


def consensus(allg: pd.DataFrame) -> pd.DataFrame:
    out = []
    for gid, g in allg.groupby("gid"):
        n = len(g)
        if n < MIN_CONSENSUS_RATERS:
            continue
        row = {"gid": gid, "id": g.id.iloc[0], "n_raters": n, "no_access_content": bool((g.no_access_content.sum() * 2) > n)}
        for d in DIMS:
            marks = [m for m in g[d] if m]
            row[d] = Counter(marks).most_common(1)[0][0] if len(marks) * 2 > n else ""
            row[f"{d}_votes"] = len(marks)
        out.append(row)
    return pd.DataFrame(out)


def fleiss(items: np.ndarray) -> float | None:
    """items: (N reviews, n raters) of category labels."""
    if len(items) < 10:
        return None
    N, n = items.shape
    cats = np.unique(items)
    counts = np.stack([(items == c).sum(1) for c in cats], 1)
    P = ((counts * (counts - 1)).sum(1) / (n * (n - 1))).mean()
    pj = counts.sum(0) / (N * n)
    Pe = (pj ** 2).sum()
    return float((P - Pe) / (1 - Pe)) if Pe < 1 else None


# ------------------------------------------------------------------ topic check
def topic_rows(doc: dict, topics: pd.DataFrame) -> pd.DataFrame:
    """Same logic as the page's topicRows(): Teresa's per-topic format for one rater."""
    groups = {g.get("name"): g for g in (doc.get("groups") or {}).values() if g.get("name")}
    saved = doc.get("topics") or {}
    rows = []
    for r in topics.itertuples():
        default_group = r.draft_parent or "Ungrouped"
        gen = default_group.startswith(GENERIC) or str(getattr(r, "generic_auto", "")).lower() == "true"
        e = {"group": default_group, "status": "invalid" if gen else "valid", "reason": "too_generic" if gen else ""}
        e.update(saved.get(str(r.Topic), {}))
        g = groups.get(e["group"])
        if e["status"] == "invalid":
            rows.append({"Topic": r.Topic, "manual_topic_status": "invalid", "manual_invalid_reason": e.get("reason", ""),
                         **{f"manual_{d}": 0 for d in DIMS}, "manual_notes": e.get("note", ""), "group": e["group"],
                         "last_updated": (g or {}).get("at", "")})
        elif g and r.Topic in (g.get("members") or []):
            rows.append({"Topic": r.Topic, "manual_topic_status": g.get("status", "valid"), "manual_invalid_reason": g.get("reason", ""),
                         **{f"manual_{d}": int((g.get("scores") or {}).get(d, 0) or 0) for d in DIMS},
                         "manual_notes": g.get("notes", ""), "group": e["group"], "last_updated": g.get("at", "")})
    return pd.DataFrame(rows)


def topic_consensus(per: dict[str, pd.DataFrame]) -> pd.DataFrame:
    allr = pd.concat([df.assign(rater=k) for k, df in per.items() if len(df)], ignore_index=True)
    out = []
    for t, g in allr.groupby("Topic"):
        n = len(g)
        inv = (g.manual_topic_status == "invalid").sum()
        if inv * 2 > n:
            reason = g.manual_invalid_reason[g.manual_topic_status == "invalid"].mode().iat[0]
            out.append({"Topic": t, "manual_topic_status": "invalid", "manual_invalid_reason": reason,
                        **{f"manual_{d}": 0 for d in DIMS}})
        else:
            k = g[g.manual_topic_status != "invalid"]
            status = "valid" if (k.manual_topic_status == "valid").sum() * 2 >= len(k) else "needs_review"
            out.append({"Topic": t, "manual_topic_status": status, "manual_invalid_reason": "",
                        **{f"manual_{d}": int(np.floor(k[f"manual_{d}"].median() + 0.5)) for d in DIMS}})
        out[-1].update({"manual_notes": f"consensus of {n} rater(s)", "n_raters": n, "n_invalid_votes": int(inv),
                        "last_updated": datetime.now().isoformat(timespec="seconds")})
    return pd.DataFrame(out).sort_values("Topic")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(C.DATA / "web_labels"))
    ap.add_argument("--labels-out", default=str(LABELS), help="for tests: write rater files somewhere else")
    ap.add_argument("--data-out", default=str(C.DATA), help="for tests: write topic scores somewhere else")
    ap.add_argument("--b-out", default=str(B_OUT), help="for tests: where Pipeline B's sentence-topic scores go")
    ap.add_argument("--csv-dir", default=None,
                    help="folder holding the page's Results-tab downloads (gold_labels_all_raters.csv, "
                         "step5_manual_scores_all_raters.csv, b2_manual_scores_all_raters.csv); use instead of --src "
                         "when nobody can pull the JSON with Claude")
    args = ap.parse_args()
    src, labels, data, b_out = Path(args.src), Path(args.labels_out), Path(args.data_out), Path(args.b_out)
    labels.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    if args.csv_dir:
        import_csv(Path(args.csv_dir), labels, data, b_out)
        return
    meta: dict = {"source": str(src), "imported_at": datetime.now().isoformat(timespec="seconds"), "gold": {}, "topics": {}}

    # gold
    frames = []
    for f in sorted((src / "gold").glob("*.json")):
        rid, doc = f.stem, body(f)
        df = gold_rows(rid, doc)
        if df.empty:
            continue
        name = safe(doc.get("code", ""), rid)
        df.drop(columns=["rater_id"]).to_csv(labels / f"{name}.csv", index=False)
        frames.append(df)
        meta["gold"][name] = {"rater_id": rid, "reviews": int(len(df))}
    if frames:
        allg = pd.concat(frames, ignore_index=True)
        allg.to_csv(labels / "gold_all_raters.csv", index=False)
        cons = consensus(allg)
        cons.to_csv(labels / "gold_consensus.csv", index=False)
        per_review = allg.groupby("gid").size()
        meta["gold_summary"] = {"raters": len(frames), "labels": int(len(allg)), "reviews_with_any": int(len(per_review)),
                                "reviews_with_5plus": int((per_review >= 5).sum()), "consensus_reviews": int(len(cons))}
        five = [g.sort_values("rater_id").head(5) for _, g in allg.groupby("gid") if len(g) >= 5]
        if five:
            meta["gold_summary"]["fleiss_kappa_discussed"] = {
                d: fleiss(np.stack([(g[d] != "").to_numpy(int) for g in five])) for d in DIMS}

    # topics: Pipeline A's whole-review topics (stage 6 reads the consensus) and Pipeline B's sentence topics (b_pipeline.py)
    sets = [("topics", C.DATA / "step5_validation_topics.csv", data, "step5_manual_scores"),
            ("topics_b2", B_OUT / "b2_validation_topics.csv", b_out, "b2_manual_scores")]
    for coll, table, dest, stem in sets:
        if not (src / coll).exists() or not table.exists():
            continue
        topics = pd.read_csv(table).fillna("")
        per = {}
        for f in sorted((src / coll).glob("*.json")):
            rid, doc = f.stem, body(f)
            if not doc.get("groups"):
                continue
            name = safe(doc.get("code", ""), rid)
            tr = topic_rows(doc, topics)
            dest.mkdir(parents=True, exist_ok=True)
            tr.to_csv(dest / f"{stem}_{name}.csv", index=False)
            per[name] = tr
            meta["topics"].setdefault(coll, {})[name] = {"rater_id": rid, "topics_scored": int(len(tr)), "groups_saved": len(doc["groups"])}
        if per:
            cons = topic_consensus(per)
            cons.to_csv(dest / f"{stem}.csv", index=False)
            meta[f"{coll}_summary"] = {"raters": len(per), "topics_with_consensus": int(len(cons)),
                                       "invalid": int((cons.manual_topic_status == "invalid").sum()),
                                       "topics_not_yet_scored": int(len(topics) - len(cons))}
    (data / "step5_import_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps({k: v for k, v in meta.items() if k.endswith("summary")}, indent=2, default=float))


def import_csv(csv_dir: Path, labels: Path, data: Path, b_out: Path) -> None:
    """The same outputs as main(), built from the CSVs the page's Results tab downloads (owner only).
    Gold: gold_labels_all_raters.csv. Topics: <stem>_all_raters.csv (one row per rater x topic)."""
    meta: dict = {"source": f"Results-tab CSVs in {csv_dir}", "imported_at": datetime.now().isoformat(timespec="seconds"),
                  "gold": {}, "topics": {}}
    gf = csv_dir / "gold_labels_all_raters.csv"
    if gf.exists():
        allg = pd.read_csv(gf, dtype={d: str for d in DIMS} | {"rater": str, "rater_id": str}).fillna({d: "" for d in DIMS})
        allg["no_access_content"] = allg.no_access_content.astype(str).str.lower().eq("true")
        allg["rater"] = allg.rater.fillna("")
        for rid, g in allg.groupby("rater_id"):
            name = safe(g.rater.iloc[0], str(rid))
            g.drop(columns=[c for c in ("rater_id", "rater_name") if c in g]).to_csv(labels / f"{name}.csv", index=False)
            meta["gold"][name] = {"rater_id": str(rid), "reviews": int(len(g))}
        allg.drop(columns=[c for c in ("rater_name",) if c in allg]).to_csv(labels / "gold_all_raters.csv", index=False)
        cons = consensus(allg)
        cons.to_csv(labels / "gold_consensus.csv", index=False)
        per_review = allg.groupby("gid").size()
        meta["gold_summary"] = {"raters": int(allg.rater_id.nunique()), "labels": int(len(allg)),
                                "reviews_with_any": int(len(per_review)), "reviews_with_5plus": int((per_review >= 5).sum()),
                                "consensus_reviews": int(len(cons))}
        five = [g.sort_values("rater_id").head(5) for _, g in allg.groupby("gid") if len(g) >= 5]
        if five:
            meta["gold_summary"]["fleiss_kappa_discussed"] = {
                d: fleiss(np.stack([(g[d] != "").to_numpy(int) for g in five])) for d in DIMS}
    for coll, dest, stem in [("topics", data, "step5_manual_scores"), ("topics_b2", b_out, "b2_manual_scores")]:
        tf = csv_dir / f"{stem}_all_raters.csv"
        if not tf.exists():
            continue
        allt = pd.read_csv(tf, dtype={"rater": str, "rater_id": str}).fillna({"rater": "", "manual_invalid_reason": "", "manual_notes": ""})
        per = {}
        dest.mkdir(parents=True, exist_ok=True)
        for rid, g in allt.groupby("rater_id"):
            name = safe(g.rater.iloc[0], str(rid))
            tr = g.drop(columns=[c for c in ("rater", "rater_name", "rater_id") if c in g])
            tr.to_csv(dest / f"{stem}_{name}.csv", index=False)
            per[name] = tr
            meta["topics"].setdefault(coll, {})[name] = {"rater_id": str(rid), "topics_scored": int(len(tr))}
        cons = topic_consensus(per)
        cons.to_csv(dest / f"{stem}.csv", index=False)
        meta[f"{coll}_summary"] = {"raters": len(per), "topics_with_consensus": int(len(cons)),
                                   "invalid": int((cons.manual_topic_status == "invalid").sum())}
    (data / "step5_import_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps({k: v for k, v in meta.items() if k.endswith("summary")}, indent=2, default=float))


if __name__ == "__main__":
    main()
