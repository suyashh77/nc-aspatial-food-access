"""Label import: majority-vote rules and the Results-tab CSV route (step5_import_web_labels.py)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step5_import_web_labels as imp  # noqa: E402
from definitions import DIMS  # noqa: E402


def _gold(rows):
    base = {"no_access_content": False, **{d: "" for d in DIMS}, "other_subject": "", "confidence": "", "notes": "",
            "seconds": "", "saved_at": ""}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_gold_consensus_needs_more_than_half_and_three_raters():
    rows = []
    for i, mark in enumerate(["+", "+", "-", "", ""]):            # 3 of 5 discuss Affordability -> kept, polarity "+"
        rows.append({"gid": 1, "id": 10, "rater": f"R{i}", "rater_id": f"u{i}", "Affordability": mark})
    for i, mark in enumerate(["+", "+", "", "", ""]):             # 2 of 5 -> dropped
        rows.append({"gid": 2, "id": 20, "rater": f"R{i}", "rater_id": f"u{i}", "Availability": mark})
    rows += [{"gid": 3, "id": 30, "rater": "R0", "rater_id": "u0", "Acceptability": "+"}]   # 1 rater -> no consensus
    c = imp.consensus(_gold(rows)).set_index("gid")
    assert c.loc[1, "Affordability"] == "+" and c.loc[1, "Affordability_votes"] == 3
    assert c.loc[2, "Availability"] == ""
    assert 3 not in c.index


def test_topic_consensus_invalid_majority_and_median_scores():
    per = {}
    for k, (status, score) in enumerate([("valid", 3), ("valid", 2), ("invalid", 0)]):
        per[f"R{k}"] = pd.DataFrame([{"Topic": 7, "manual_topic_status": status, "manual_invalid_reason": "" if status == "valid" else "too_generic",
                                      **{f"manual_{d}": score if d == "Affordability" else 0 for d in DIMS}}])
    c = imp.topic_consensus(per).iloc[0]
    assert c.manual_topic_status == "valid" and c.manual_Affordability == 3       # median of 3, 2 = 2.5, rounded half up
    assert c.n_invalid_votes == 1


def test_csv_route_writes_the_files_stage6_and_decision_tests_read(tmp_path):
    src = tmp_path / "csv"
    src.mkdir()
    rows = [{"gid": g, "id": 100 + g, "rater": f"R{i}", "rater_name": "", "rater_id": f"u{i}", "Accessibility": "+" if i < 3 else ""}
            for g in range(3) for i in range(5)]
    _gold(rows).to_csv(src / "gold_labels_all_raters.csv", index=False)
    trows = [{"rater": f"R{i}", "rater_name": "", "rater_id": f"u{i}", "Topic": t,
              "manual_topic_status": "invalid" if t == 0 else "valid", "manual_invalid_reason": "too_generic" if t == 0 else "",
              **{f"manual_{d}": 1 for d in DIMS}, "manual_notes": "", "group": "g", "last_updated": ""}
             for t in range(3) for i in range(3)]
    pd.DataFrame(trows).to_csv(src / "step5_manual_scores_all_raters.csv", index=False)
    labels, data, b = tmp_path / "labels", tmp_path / "data", tmp_path / "b"
    labels.mkdir(), data.mkdir()
    imp.import_csv(src, labels, data, b)
    cons = pd.read_csv(labels / "gold_consensus.csv").fillna("")
    assert len(cons) == 3 and (cons.Accessibility == "+").all()
    s = pd.read_csv(data / "step5_manual_scores.csv")
    assert set(s.Topic[s.manual_topic_status == "valid"]) == {1, 2}
    assert (labels / "R0.csv").exists() and (data / "step5_import_meta.json").exists()
