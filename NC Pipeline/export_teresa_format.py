"""Export one NC county's text reviews in the format of Teresa's grocery_store_reviews_text.csv.

    uv run --no-project --with pandas --with xlsxwriter python export_teresa_format.py HAYWOOD

Writes to data/exports/<county>/: grocery_store_reviews_text.csv (text reviews only, her 44 columns in
her order, the file her notebooks read), <county>_stores.csv, and <county>_reviews_and_stores.xlsx (every
review, with or without text, plus the stores). Fields the NC database doesn't hold are left blank.

The stores table has no county. A store's county comes from its coordinates inside the block-group
polygons in the grocery map data (98.3% agreement with the 2,180 stores whose county is known), then
from the retail frame when the coordinates fall outside every polygon.
"""
from __future__ import annotations

import json
import sqlite3
import sys

import numpy as np
import pandas as pd

import config as C

FRAME_DB = C.ROOT / "Reviews Data" / "Step_1_NC_Food_Retail_Frame.db"
BLOCK_GROUPS = C.ROOT / "grocery-foot-traffic" / "map" / "data" / "bg.geojson"
OUT = C.DATA / "exports"

COLUMNS = ["address", "categoryName", "cid", "city", "countryCode", "error", "fid", "isAdvertisement", "isLocalGuide",
           "language", "likesCount", "location/lat", "location/lng", "name", "originalLanguage", "originalText",
           "permanentlyClosed", "placeId", "postalCode", "publishAt", "publishedAtDate", "responseFromOwnerDate",
           "responseFromOwnerText", "reviewId", "reviewOrigin", "reviewUrl", "reviewerId", "reviewerNumberOfReviews",
           "reviewerPhotoUrl", "reviewerUrl", "reviewsCount", "scrapedAt", "searchString", "stars", "state", "storeNumber",
           "street", "temporarilyClosed", "text", "title", "totalScore", "translated", "translatedLanguage", "url"]


def read_only(path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _in_ring(x: float, y: float, ring: np.ndarray) -> bool:
    xa, ya = ring[:, 0], ring[:, 1]
    xb, yb = np.roll(xa, 1), np.roll(ya, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossings = ((ya > y) != (yb > y)) & (x < (xb - xa) * (y - ya) / (yb - ya) + xa)
    return bool(crossings.sum() % 2)


def load_polygons() -> tuple[list, np.ndarray]:
    polys = []
    for f in json.loads(BLOCK_GROUPS.read_text())["features"]:
        geo = f["geometry"]
        for poly in [geo["coordinates"]] if geo["type"] == "Polygon" else geo["coordinates"]:
            rings = [np.asarray(r)[:, :2] for r in poly]
            polys.append((f["properties"]["county"].upper().replace(" ", ""), rings))
    boxes = np.array([[r[0][:, 0].min(), r[0][:, 0].max(), r[0][:, 1].min(), r[0][:, 1].max()] for _, r in polys])
    return polys, boxes


def county_of(x: float, y: float, polys: list, boxes: np.ndarray) -> str | None:
    hits = (boxes[:, 0] <= x) & (x <= boxes[:, 1]) & (boxes[:, 2] <= y) & (y <= boxes[:, 3])
    for i in np.where(hits)[0]:
        name, rings = polys[i]
        if _in_ring(x, y, rings[0]) and not any(_in_ring(x, y, h) for h in rings[1:]):
            return name
    return None


def cid_from_fid(fid) -> str:
    """Google's cid is the decimal form of the second half of the feature id ("0x...:0x...")."""
    if not isinstance(fid, str) or ":" not in fid:
        return ""
    return str(int(fid.split(":")[1], 16))


def stores_in_county(db: sqlite3.Connection, county: str) -> pd.DataFrame:
    stores = pd.read_sql("SELECT * FROM stores", db)
    with read_only(FRAME_DB) as frame:
        known = pd.read_sql("SELECT place_id, county AS frame_county FROM stores WHERE county <> '' AND place_id IS NOT NULL", frame)
    known = known.drop_duplicates("place_id")
    known["frame_county"] = known.frame_county.str.upper().str.replace(" ", "")
    stores = stores.merge(known, how="left", on="place_id")
    polys, boxes = load_polygons()
    stores["polygon_county"] = [county_of(x, y, polys, boxes) for x, y in zip(stores.source_lon, stores.source_lat)]
    stores["county"] = stores.polygon_county.fillna(stores.frame_county)
    return stores[stores.county == county].copy()


def to_teresa(reviews: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    d = reviews.merge(stores, on="store_id", how="left", validate="many_to_one", suffixes=("", "_store"))
    out = pd.DataFrame(index=d.index, columns=COLUMNS, dtype=object)
    out["address"] = d.full_address
    out["categoryName"] = d.category
    out["cid"] = d.feature_id.map(cid_from_fid)
    out["city"] = d.city
    out["countryCode"] = "US"
    out["fid"] = d.feature_id
    out["isLocalGuide"] = d.is_local_guide.map({1: True, 0: False})
    out["likesCount"] = d.helpful_count
    out["location/lat"] = d.source_lat
    out["location/lng"] = d.source_lon
    out["name"] = d.reviewer_name
    out["permanentlyClosed"] = d.business_status.eq("CLOSED_PERMANENTLY")
    out["placeId"] = d.place_id
    out["postalCode"] = d.zip
    out["publishAt"] = d.date_relative
    # A few older rows have no Google review id; her notebooks need a unique, non-empty one.
    out["reviewId"] = d.review_id.where(d.review_id.notna() & d.review_id.ne(""), "nc_db_" + d.id.astype(str))
    out["reviewOrigin"] = "Google"
    out["reviewerNumberOfReviews"] = d.reviewer_review_count
    out["reviewsCount"] = d.target_reviews
    out["scrapedAt"] = d.scraped_at
    out["stars"] = d.rating
    out["state"] = "North Carolina"
    out["storeNumber"] = d.store_id
    out["street"] = d.street
    out["temporarilyClosed"] = d.business_status.eq("CLOSED_TEMPORARILY")
    out["text"] = d.review_text
    out["title"] = d.outscraper_name.fillna(d.store_name)
    out["totalScore"] = d.rating_store
    out["translated"] = 0
    return out


def write_excel(county: str, stores: pd.DataFrame, summary: pd.DataFrame, target) -> None:
    """One workbook: every review of the county (with and without text) in her columns, and one row per store."""
    with read_only(C.REVIEWS_DB) as db:
        ids = ",".join(str(int(i)) for i in stores.store_id)
        everything = pd.read_sql(f"SELECT * FROM reviews WHERE store_id IN ({ids})", db)
    reviews = to_teresa(everything, stores).drop_duplicates("reviewId")
    totals = everything.groupby("store_id").size().rename("all_reviews")
    per_store = summary.merge(totals, how="left", left_on="store_id", right_index=True).fillna({"all_reviews": 0})
    per_store["all_reviews"] = per_store.all_reviews.astype(int)
    keep = ["store_id", "place_id", "feature_id", "outscraper_name", "category", "full_address", "city", "zip", "source_lat", "source_lon",
            "rating", "target_reviews", "business_status", "scrape_status", "county", "all_reviews", "text_reviews"]
    per_store = per_store[keep].sort_values("all_reviews", ascending=False)
    # Review text is written as plain text: no formulas or links, whatever a reviewer typed.
    options = {"strings_to_formulas": False, "strings_to_urls": False, "strings_to_numbers": False}
    path = target / f"{county.lower()}_reviews_and_stores.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter", engine_kwargs={"options": options}) as xl:
        reviews.to_excel(xl, sheet_name="Reviews", index=False, freeze_panes=(1, 0))
        per_store.to_excel(xl, sheet_name="Stores", index=False, freeze_panes=(1, 0))
    print(f"{path.name}: {len(reviews):,} reviews, {len(per_store)} stores")


def main(county: str) -> None:
    county = county.upper().replace(" ", "")
    target = OUT / county.lower()
    target.mkdir(parents=True, exist_ok=True)
    with read_only(C.REVIEWS_DB) as db:
        stores = stores_in_county(db, county)
        if stores.empty:
            raise SystemExit(f"No stores found for county {county!r}")
        ids = ",".join(str(int(i)) for i in stores.store_id)
        reviews = pd.read_sql(f"SELECT * FROM reviews WHERE store_id IN ({ids}) AND review_text IS NOT NULL AND TRIM(review_text) <> ''", db)
    out = to_teresa(reviews, stores).drop_duplicates("reviewId")  # a few reviews were stored twice under one id
    assert list(out.columns) == COLUMNS and out.reviewId.is_unique and out.text.notna().all()
    out.to_csv(target / "grocery_store_reviews_text.csv", index=False)
    counts = reviews.groupby("store_id").size().rename("text_reviews")
    summary = stores.merge(counts, how="left", left_on="store_id", right_index=True).fillna({"text_reviews": 0})
    summary["text_reviews"] = summary.text_reviews.astype(int)
    summary[["store_id", "outscraper_name", "category", "city", "business_status", "frame_county", "polygon_county", "text_reviews"]] \
        .sort_values("text_reviews", ascending=False).to_csv(target / f"{county.lower()}_stores.csv", index=False)
    write_excel(county, stores, summary, target)
    print(f"{county}: {len(stores)} stores, {len(out):,} text reviews -> {target}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "HAYWOOD")
