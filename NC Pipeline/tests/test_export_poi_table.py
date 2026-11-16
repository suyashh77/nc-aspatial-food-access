"""The v2 store table: one row per master store, v1's 28 columns, review counts that add up to freeze v2."""
from export_poi_table import COLS, build


def test_poi_table_matches_freeze_v2():
    t, dropped = build()
    assert list(t.columns) == COLS
    assert t.master_store_id.is_unique
    assert len(t) == 4442
    assert int(t.reviews_scraped.sum()) == 1_941_538
    assert t.loc[t.master_store_id == 9472, "latitude"].isna().all()
