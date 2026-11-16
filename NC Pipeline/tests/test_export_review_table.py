"""export_review_table.year_approx: Google's relative dates -> approximate review year."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_review_table import year_approx  # noqa: E402


def test_year_approx():
    rel = pd.Series(["7 years ago", "Edited a year ago", "3 months ago", "a week ago", "an hour ago", "", None])
    got = year_approx(rel, pd.Series(["2026-06-01 10:00:00"] * 7)).tolist()
    assert got[:5] == [2019, 2025, 2026, 2026, 2026]
    assert pd.isna(got[5]) and pd.isna(got[6])
