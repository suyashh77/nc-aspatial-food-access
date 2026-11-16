"""Rebuild the store table (NC_Grocery_POIs.xlsx) from freeze v2.

    .venv/bin/python export_poi_table.py

The table handed around so far (grocery-foot-traffic/NC_Grocery_POIs.xlsx; Reviews Data/HANDOFF.md item 2) was built
from freeze v1 (4,447-4,452 rows). This writes the same 28 columns, one row per store in the master DB (= freeze v2,
4,442 stores; checked against config.REVIEWS_DB counts), with v2's own status, counts and coordinates. The Dewey match
columns, which the master DB doesn't hold, are carried over from the v1 table by store id.

Output: data/exports/NC_Grocery_POIs_v2.xlsx (+ .csv), and a list of v1 rows not in v2.
The master DB is opened read-only. Store 9472's coordinates stay blank (accepted flag: Google's pin is outside NC).
"""
import sqlite3

import pandas as pd

import config as C

V1 = C.ROOT / "grocery-foot-traffic" / "NC_Grocery_POIs.xlsx"
OUT = C.DATA / "exports"
COLS = ['master_store_id', 'store_name', 'business_name', 'street', 'city', 'state', 'zip', 'full_address', 'latitude', 'longitude',
        'category', 'type', 'subtypes', 'google_place_id', 'google_feature_id', 'phone', 'google_rating', 'business_status',
        'target_reviews', 'reviews_scraped', 'scrape_status', 'status_reason', 'dewey_placekey', 'dewey_brand', 'dewey_chain',
        'dewey_top_category', 'dewey_naics', 'dewey_match_grade']
DEWEY = [c for c in COLS if c.startswith("dewey_")]
NO_COORDS = {9472}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(f"file:{C.REVIEWS_DB}?mode=ro", uri=True)
    s = pd.read_sql("SELECT * FROM stores", con)
    n_reviews = pd.read_sql("SELECT store_id, COUNT(*) AS n FROM reviews GROUP BY store_id", con).set_index("store_id").n
    t = pd.DataFrame({
        "master_store_id": s.store_id, "store_name": s.store_name, "business_name": s.business_name, "street": s.street,
        "city": s.city, "state": s.state, "zip": s.zip, "full_address": s.full_address,
        "latitude": s.source_lat, "longitude": s.source_lon, "category": s.category, "type": s.type, "subtypes": s.subtypes,
        "google_place_id": s.place_id, "google_feature_id": s.feature_id, "phone": s.phone, "google_rating": s.rating,
        "business_status": s.business_status, "target_reviews": s.target_reviews,
        "reviews_scraped": s.store_id.map(n_reviews).fillna(0).astype(int), "scrape_status": s.scrape_status,
        "status_reason": s.status_reason})
    t.loc[t.master_store_id.isin(NO_COORDS), ["latitude", "longitude"]] = None
    v1 = pd.read_excel(V1)
    t = t.merge(v1[["master_store_id", *DEWEY]], on="master_store_id", how="left")[COLS]
    dropped = v1[~v1.master_store_id.isin(t.master_store_id)][["master_store_id", "store_name", "city", "scrape_status"]]
    return t.sort_values("master_store_id"), dropped


def main():
    t, dropped = build()
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_excel(OUT / "NC_Grocery_POIs_v2.xlsx", index=False, sheet_name="NC_Grocery_POIs")
    t.to_csv(OUT / "NC_Grocery_POIs_v2.csv", index=False)
    dropped.to_csv(OUT / "NC_Grocery_POIs_v1_rows_not_in_v2.csv", index=False)
    print(f"{len(t):,} stores, {int(t.reviews_scraped.sum()):,} reviews; status: {t.scrape_status.value_counts().to_dict()}")
    print(f"{len(dropped)} v1 rows not in v2; Dewey placekey on {t.dewey_placekey.notna().sum():,} stores")


if __name__ == "__main__":
    main()
