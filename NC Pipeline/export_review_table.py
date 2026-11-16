"""The analysis export: one row per review (all 1,941,538), joined to its store, with every pipeline's labels.

    .venv/bin/python export_review_table.py            # about 2 minutes

Writes data/exports/NC_reviews_with_dimensions.parquet and a 2,000-row preview CSV next to it. Re-run it after the
human labels are in (run_after_labeling.sh does): the label columns follow whatever validity the pipelines used.

Columns
  ids        review_db_id (row id in the master DB), review_id (Google's), store_id, place_id
  store      store_name, category, city, county, lat, lon
  review     stars (1-5; 636 have none), has_text, text, date_relative (Google's words, e.g. "Edited 3 years ago"),
             edited, review_year_approx (scrape year minus the relative age; at best +-1 year), scraped_at,
             has_owner_response, is_local_guide, reviewer_review_count
  flags      store_scrape_status (complete / incomplete), store_target_reviews (Google's live count),
             store_holds_newest_only (target over 1,500: the store keeps only its newest reviews),
             store_target_exactly_1500 (an old capped target, so the true size is unknown)
  Pipeline A (NC refit)   a_topic (-1 = outlier), a_dim1..a_dim3, a_topic_valid
  Pipeline A (frozen)     frozen_topic, frozen_dim1..frozen_dim3, frozen_topic_valid
  Pipeline B              b_<Dimension>: mean text sentiment (-1..+1) of the review's sentences about that dimension;
                          empty = the review doesn't count toward it
  labels_status           "provisional" until the human topic check is imported, then "human"
Reviewer names are left out on purpose.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import config as C
from definitions import DIMS
from export_teresa_format import read_only
from step6_store_scores import stores_table, validity

B_OUT = C.ROOT / "NC Pipeline Improved" / "out"
OUT = C.DATA / "exports"


def year_approx(date_relative: pd.Series, scraped_at: pd.Series) -> pd.Series:
    s = date_relative.fillna("").str.replace("Edited ", "", regex=False)
    n = s.str.extract(r"^(\d+|a|an) (\w+?)s? ago$")
    num = pd.to_numeric(n[0].replace({"a": "1", "an": "1"}), errors="coerce")
    years = num * n[1].map({"year": 1.0, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365, "hour": 0.0, "minute": 0.0})
    scraped = pd.to_datetime(scraped_at, errors="coerce")
    return np.floor(scraped.dt.year + scraped.dt.dayofyear / 365.25 - years).astype("Int64")


def main() -> None:
    with read_only(C.REVIEWS_DB) as db:
        rv = pd.read_sql("SELECT id AS review_db_id, review_id, store_id, rating AS stars, review_text AS text, date_relative, "
                         "scraped_at, has_owner_response, is_local_guide, reviewer_review_count FROM reviews", db)
        flags = pd.read_sql("SELECT store_id, scrape_status AS store_scrape_status, target_reviews AS store_target_reviews FROM stores", db)
    st = stores_table()[["store_id", "place_id", "store_name", "category", "city", "county", "lat", "lon"]]
    rv["has_text"] = rv.text.fillna("").str.strip().ne("")
    rv["edited"] = rv.date_relative.fillna("").str.startswith("Edited")
    rv["review_year_approx"] = year_approx(rv.date_relative, rv.scraped_at)
    flags["store_holds_newest_only"] = flags.store_target_reviews > 1500
    flags["store_target_exactly_1500"] = flags.store_target_reviews == 1500
    rv = rv.merge(st, on="store_id", how="left").merge(flags, on="store_id", how="left")

    # Pipeline A, both arms (row-aligned with step1_reviews.parquet, keyed by review_db_id)
    for arm, f, pre in (("nc", "step4_review_dims.parquet", "a"), ("frozen", "step4_review_dims_frozen.parquet", "frozen")):
        valid, _ = validity(arm)
        d = pd.read_parquet(C.DATA / f, columns=["id", "topic", "pred_dim", "pred_dim2", "pred_dim3"])
        d = d.rename(columns={"id": "review_db_id", "topic": f"{pre}_topic", "pred_dim": f"{pre}_dim1",
                              "pred_dim2": f"{pre}_dim2", "pred_dim3": f"{pre}_dim3"})
        d[f"{pre}_topic_valid"] = d[f"{pre}_topic"].isin(valid)
        rv = rv.merge(d, on="review_db_id", how="left")

    # Pipeline B (b_review_dims.parquet: row = step1 row)
    ids = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id"]).id.to_numpy()
    bd = pd.read_parquet(B_OUT / "b_review_dims.parquet")
    bw = bd.pivot(index="row", columns="dim", values="text").reindex(columns=DIMS)
    bw.columns = [f"b_{d}" for d in DIMS]
    bw.insert(0, "review_db_id", ids[bw.index.to_numpy()])
    rv = rv.merge(bw.reset_index(drop=True), on="review_db_id", how="left")

    human = (C.DATA / "step5_manual_scores.csv").exists()
    rv["labels_status"] = "human" if human else "provisional"
    cols = ["review_db_id", "review_id", "store_id", "place_id", "store_name", "category", "city", "county", "lat", "lon",
            "stars", "has_text", "text", "date_relative", "edited", "review_year_approx", "scraped_at", "has_owner_response",
            "is_local_guide", "reviewer_review_count", "store_scrape_status", "store_target_reviews", "store_holds_newest_only",
            "store_target_exactly_1500", "a_topic", "a_dim1", "a_dim2", "a_dim3", "a_topic_valid", "frozen_topic",
            "frozen_dim1", "frozen_dim2", "frozen_dim3", "frozen_topic_valid", *[f"b_{d}" for d in DIMS], "labels_status"]
    rv = rv[cols].sort_values("review_db_id")
    OUT.mkdir(parents=True, exist_ok=True)
    rv.to_parquet(OUT / "NC_reviews_with_dimensions.parquet", index=False)
    rv[rv.has_text].sample(2000, random_state=0).sort_values("review_db_id").to_csv(OUT / "NC_reviews_with_dimensions_preview.csv", index=False)
    summary = {"reviews": len(rv), "with_text": int(rv.has_text.sum()), "stores": int(rv.store_id.nunique()),
               "with_a_label": int(rv.a_dim1.notna().sum()), "a_valid_and_labelled": int((rv.a_topic_valid & rv.a_dim1.notna()).sum()),
               "with_b_label": int(rv[[f"b_{d}" for d in DIMS]].notna().any(axis=1).sum()), "labels_status": rv.labels_status.iat[0],
               "year_approx_missing": int(rv.review_year_approx.isna().sum())}
    (OUT / "NC_reviews_with_dimensions_meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
