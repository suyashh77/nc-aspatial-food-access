"""Step 7: census-tract access scores and the equity regression (thesis section 4.1.6, eqs. 4.19-4.21).

    .venv/bin/python step7_tracts.py                      # all NC tracts, NC-arm store scores
    .venv/bin/python step7_tracts.py --arm frozen
    .venv/bin/python step7_tracts.py --county HAYWOOD

  w_il = exp(-D_il / h), h = 1 mile, D = straight-line miles from the tract's internal point to store l   (4.19)
  E_ij = sum_l w_il z_lj / sum_l w_il   over stores with a score on dimension j                         (4.20)
  Y_i = b0 + b1 pctBlack_c + b2 income_c + b3 pctBlack_c x income_c, weighted by tract population         (4.21)
  Y = each E_j, and distance to the nearest store (z-scored). Covariates centred (population-weighted mean).
Units: pctBlack as a share 0-1 and income in $10,000s. These reproduce her Table 5.3 best (decisions/d9).

Census inputs (data_external/census/, downloaded from www2.census.gov, no API key):
  ACS 2022 5-year B01003 (population), B02001 (race: Black alone = B02001_E003), B19013 (median household income),
  2022 Gazetteer tract internal points.
Outputs (data/): step7_tracts[_tag].csv, step7_regressions[_tag].csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config as C
from definitions import DIMS

MILES_PER_KM = 0.621371


def acs(table: str, cols: dict) -> pd.DataFrame:
    d = pd.read_csv(C.CENSUS / f"acsdt5y2022-{table}.dat", sep="|", dtype=str)
    d = d[d.GEO_ID.str.startswith("1400000US37")]
    out = pd.DataFrame({"GEOID": d.GEO_ID.str[9:]})
    for k, v in cols.items():
        x = pd.to_numeric(d[k], errors="coerce")
        out[v] = x.where(x >= 0).to_numpy()
    return out


def nc_tracts() -> pd.DataFrame:
    g = pd.read_csv(C.CENSUS / "2022_Gaz_tracts_national.txt", sep="\t", dtype={"GEOID": str})
    g.columns = [c.strip() for c in g.columns]
    g = g[g.GEOID.str.startswith("37")][["GEOID", "INTPTLAT", "INTPTLONG", "ALAND_SQMI"]]
    g.columns = ["GEOID", "lat", "lon", "land_sqmi"]
    t = (g.merge(acs("b01003", {"B01003_E001": "pop"}), on="GEOID")
          .merge(acs("b02001", {"B02001_E003": "black"}), on="GEOID")
          .merge(acs("b19013", {"B19013_E001": "income"}), on="GEOID"))
    t["pct_black"] = t.black / t["pop"]
    t["county_fips"] = t.GEOID.str[2:5]
    return t


def miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine distance matrix in miles (tracts x stores)."""
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a)) * MILES_PER_KM


def tract_scores(tracts: pd.DataFrame, stores: pd.DataFrame, zcol: str = "{d}_pooled_z", h: float = C.DECAY_H_MILES) -> pd.DataFrame:
    D = miles(tracts.lat.to_numpy(), tracts.lon.to_numpy(), stores.lat.to_numpy(), stores.lon.to_numpy())
    W = np.exp(-D / h)
    out = tracts.copy()
    out["dist_nearest_store_mi"] = D.min(1)
    for d in DIMS:
        z = stores[zcol.format(d=d)].to_numpy(float)
        ok = np.isfinite(z)
        num, den = (W[:, ok] * z[ok]).sum(1), W[:, ok].sum(1)
        out[f"E_{d}"] = np.where(den > 1e-300, num / np.where(den > 1e-300, den, 1), np.nan)
    return out


def wls(t: pd.DataFrame, y: str, income_unit: float = 10_000.0, black_unit: float = 1.0) -> dict:
    import statsmodels.api as sm
    df = t[[y, "pct_black", "income", "pop"]].replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["pop"] > 0]
    w = df["pop"]
    xb = df.pct_black / black_unit
    xi = df.income / income_unit
    xb, xi = xb - np.average(xb, weights=w), xi - np.average(xi, weights=w)
    X = sm.add_constant(pd.DataFrame({"pct_black_c": xb, "income_c": xi, "black_x_income": xb * xi}))
    r = sm.WLS(df[y], X, weights=w).fit()
    return {"outcome": y, "n_tracts": int(len(df)), **{f"b_{k}": r.params[k] for k in X.columns},
            **{f"p_{k}": r.pvalues[k] for k in X.columns[1:]}, "r2": r.rsquared}


def run(tracts, stores, zcol="{d}_pooled_z", **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = tract_scores(tracts, stores, zcol)
    t["dist_nearest_z"] = (t.dist_nearest_store_mi - t.dist_nearest_store_mi.mean()) / t.dist_nearest_store_mi.std()
    regs = [wls(t, "dist_nearest_z", **kw)] + [wls(t, f"E_{d}", **kw) for d in DIMS]
    return t, pd.DataFrame(regs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="nc", choices=["nc", "frozen"])
    ap.add_argument("--county", default=None, help="restrict tracts to one county (stores: all NC, so border stores count)")
    args = ap.parse_args()
    stores = pd.read_csv(C.DATA / ("step6_store_scores.csv" if args.arm == "nc" else "step6_store_scores_frozen.csv"))
    tracts = nc_tracts()
    tag = args.arm
    if args.county:
        from step6_store_scores import stores_table
        st = stores_table()
        fips = pd.read_csv(C.CENSUS / "2022_Gaz_tracts_national.txt", sep="\t", dtype={"GEOID": str})  # noqa: F841 (kept simple)
        cty = args.county.upper().replace(" ", "")
        lat, lon = st[st.county == cty].lat.mean(), st[st.county == cty].lon.mean()
        # tracts of the county: the county FIPS most common among tracts nearest its stores
        near = tracts.assign(d=miles(tracts.lat.to_numpy(), tracts.lon.to_numpy(), np.array([lat]), np.array([lon]))[:, 0])
        f = near.nsmallest(5, "d").county_fips.mode().iat[0]
        tracts = tracts[tracts.county_fips == f]
        tag += f"_{cty.lower()}"
    t, regs = run(tracts, stores)
    t.to_csv(C.DATA / f"step7_tracts_{tag}.csv", index=False)
    regs.to_csv(C.DATA / f"step7_regressions_{tag}.csv", index=False)
    print(regs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
