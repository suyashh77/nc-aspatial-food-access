"""Raleigh case study: Teresa's own results next to Pipeline A (both arms) and Pipeline B, for Caroline's presentation.

    ../NC\\ Pipeline/.venv/bin/python b_raleigh.py      (after b_pipeline.py and b6_local_market.py --scores b)

Stores: her thesis store_scores.csv (49 stores, matched to ours by place id, then Google feature id, as in D8) against
  A frozen (her model on our scrape), A NC refit, B (statewide model with county and category levels, text sentiment)
  and B's person-centred local z (16-minute markets). Spearman rank agreement with her pooled scores, per dimension.
Tracts: the Wake tract set that reproduced her Table 5.3 best in D9 (tracts whose nearest of her stores is <= 1 mile),
  and all Wake tracts. Same population-weighted regression; her coefficients, A (stage 7, statewide stores, exp(-miles))
  and B6 (mean of the stores within a 16-minute drive, B's scores).
Also: all Wake tracts with 8-minute markets (16 minutes reach most of Wake from anywhere), A and B store scores.
Output: out/b_raleigh.json, out/b_raleigh_stores.csv
"""
from common_b import *
from step6_store_scores import stores_table
from step7_tracts import miles, nc_tracts, wls
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--weights", choices=["visits", "equal"], default="visits", help="foot-traffic weights (optional Dewey data) or equal")
WEIGHTS = ap.parse_args().weights
SFX = "" if WEIGHTS == "visits" else "_equal"

HER_TABLE53 = {"dist_nearest_z": (4.078, 0.204, 0.680, 0.358), "E_Availability": (-0.479, -0.016, -0.024, 0.063),
               "E_Accessibility": (-0.238, 0.005, 0.016, 0.028), "E_Accommodation": (-0.149, 0.020, 0.174, 0.100),
               "E_Affordability": (1.640, 0.041, 0.062, 0.258), "E_Acceptability": (-0.123, 0.099, 0.210, 0.415)}
R = {}
her = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/store_scores.csv")
th = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/raleigh_ring_reviews_with_dims.csv", low_memory=False,
                 usecols=["placeId", "fid"]).drop_duplicates("placeId")
her = her.merge(th, on="placeId", how="left")
st = stores_table()
m = pd.concat([her.merge(st[["store_id", "place_id"]], left_on="placeId", right_on="place_id"),
               her.merge(st[["store_id", "feature_id"]], left_on="fid", right_on="feature_id")]).drop_duplicates("placeId")
m = m.set_index("store_id")
A_fz = pd.read_csv(C.DATA / "step6_store_scores_raleigh_frozen.csv").set_index("store_id")
A_nc = pd.read_csv(C.DATA / "step6_store_scores_raleigh.csv").set_index("store_id")
B = pd.read_csv(OUT / "b_store_scores.csv").set_index("store_id")
LZ = pd.read_csv(OUT / f"b6_store_local_z_b{SFX}.csv").set_index("store_id")
out = pd.DataFrame(index=m.index)
out["name"] = st.set_index("store_id").name.reindex(m.index)
R["stores_matched"] = int(len(m))
R["store_agreement_with_teresa"] = {}
for d in DIMS:
    cols = {"teresa": m[f"{d}_pooled_mean"], "A_frozen": A_fz[f"{d}_pooled_mean_signal"], "A_nc": A_nc[f"{d}_pooled_mean_signal"],
            "B": B[f"{d}_b_mean"], "B_stars": B[f"{d}_bstars_mean"], "B_local_z": LZ[d]}
    for k, v in cols.items():
        out[f"{d}_{k}"] = v.reindex(m.index)
    row = {}
    for k in cols:
        if k == "teresa":
            continue
        j = out[[f"{d}_teresa", f"{d}_{k}"]].dropna()
        row[k] = round(float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")), 2) if len(j) > 5 else None
    row["stores"] = int(out[[f"{d}_teresa", f"{d}_B"]].dropna().shape[0])
    name = lambda c: [out.name[i] for i in out[c].dropna().nlargest(3).index]
    row["top3_teresa"], row["top3_B"] = name(f"{d}_teresa"), name(f"{d}_B")
    R["store_agreement_with_teresa"][d] = row
    print(d, {k: v for k, v in row.items() if not k.startswith("top")}, flush=True)
out.reset_index().to_csv(OUT / f"b_raleigh_stores{SFX}.csv", index=False)

# tracts
T = nc_tracts()
wake = T[T.county_fips == "183"].copy()
D = miles(wake.lat.to_numpy(), wake.lon.to_numpy(), her["location/lat"].to_numpy(), her["location/lng"].to_numpy())
sets = {"within_1mi_of_her_stores": wake[D.min(1) <= 1].GEOID, "all_wake": wake.GEOID}
A7 = pd.read_csv(C.DATA / "step7_tracts_nc.csv", dtype={"GEOID": str}).set_index("GEOID")
B6 = pd.read_csv(OUT / f"b6_tracts_b{SFX}.csv", dtype={"GEOID": str}).set_index("GEOID")
R["tracts"] = {}
for sname, ids in sets.items():
    res = {"n_tracts": int(len(ids)), "people_no_store_16min": int(B6.loc[B6.index.intersection(ids)].query("n_stores_16 == 0")["pop"].sum())}
    for d in DIMS:
        a = wls(A7.loc[A7.index.intersection(ids)].reset_index(), f"E_{d}")
        b = wls(B6.loc[B6.index.intersection(ids)].reset_index(), f"E_{d}_16")
        hb = HER_TABLE53[f"E_{d}"]
        res[d] = {"teresa_b_black": hb[0], "teresa_r2": hb[3],
                  "A_b_black": round(a["b_pct_black_c"], 3), "A_p": round(a["p_pct_black_c"], 3), "A_r2": round(a["r2"], 3),
                  "B_b_black": round(b["b_pct_black_c"], 3), "B_p": round(b["p_pct_black_c"], 3), "B_r2": round(b["r2"], 3)}
    R["tracts"][sname] = res
    print(sname, json.dumps(res)[:600], flush=True)
# shorter markets for a city: 8-minute drive (B6's 16 minutes reach most of Wake from anywhere)
dt = pd.read_parquet(OUT / "b6_drive_times.parquet")
if WEIGHTS == "visits":
    ft = pd.read_csv(C.ROOT / "grocery-foot-traffic/stores.csv", usecols=["master_store_id", "category", "visits_per_week_latest12"]).set_index("master_store_id")
    v = ft.visits_per_week_latest12.fillna(ft.category.map(ft.groupby("category").visits_per_week_latest12.median())).fillna(ft.visits_per_week_latest12.median())
A_state = pd.read_csv(C.DATA / "step6_store_scores.csv").set_index("store_id")
P = dt[dt.GEOID.isin(wake.GEOID) & (dt.minutes <= 8)]
P = P.assign(v=P.store_id.map(v).fillna(v.median())) if WEIGHTS == "visits" else P.assign(v=1.0)
tw = wake.set_index("GEOID")
R["tracts_8min"] = {"n_tracts_with_store": int(P.GEOID.nunique()), "median_stores_within_8": float(P.groupby("GEOID").size().median()),
                    "median_stores_within_16": float(dt[dt.GEOID.isin(wake.GEOID) & (dt.minutes <= 16)].groupby("GEOID").size().median())}
for d in DIMS:
    row = {}
    for name, sc in (("A", A_state[f"{d}_pooled_mean_signal"]), ("B", B[f"{d}_b_mean"])):
        q = P.assign(s=P.store_id.map(sc)).dropna(subset=["s"])
        g = q.assign(vs=q.v * q.s).groupby("GEOID")
        tw[f"E8_{name}_{d}"] = (g.vs.sum() / g.v.sum())
        x = tw[f"E8_{name}_{d}"]
        ok = x.notna() & (tw["pop"] > 0)
        m_ = np.average(x[ok], weights=tw["pop"][ok])
        tw[f"E8_{name}_{d}"] = (x - m_) / np.sqrt(np.average((x[ok] - m_) ** 2, weights=tw["pop"][ok]))
        r = wls(tw.reset_index(), f"E8_{name}_{d}")
        row[name] = {"b_black": round(r["b_pct_black_c"], 3), "p": round(r["p_pct_black_c"], 3), "r2": round(r["r2"], 3), "n": r["n_tracts"]}
    R["tracts_8min"][d] = row
    print("8min", d, row, flush=True)
R["store_weights"] = WEIGHTS
save(f"b_raleigh{SFX}", R)
