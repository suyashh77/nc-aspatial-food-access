"""D9. Reproduce Teresa's Table 5.3 (tract regressions) from her own store scores, to pin down stage 7's unstated
choices: which z (raw-mean z in her data file, or pooled z per thesis eq. 4.18), which tracts, which units."""
import json
from common import *
import step7_tracts as S7

TARGET = {"dist_nearest_z": (4.078, 0.204, 0.680, 0.358), "E_Availability": (-0.479, -0.016, -0.024, 0.063),
          "E_Accessibility": (-0.238, 0.005, 0.016, 0.028), "E_Accommodation": (-0.149, 0.020, 0.174, 0.100),
          "E_Affordability": (1.640, 0.041, 0.062, 0.258), "E_Acceptability": (-0.123, 0.099, 0.210, 0.415)}
her = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/store_scores.csv").rename(columns={"location/lat": "lat", "location/lng": "lon"})
for d in DIMS:
    pmn = her[f"{d}_pooled_mean"]
    her[f"{d}_pooled_z"] = (pmn - pmn.mean()) / pmn.std()
    her[f"{d}_raw_z"] = her[f"{d}_z"]
T = S7.nc_tracts()
wake = T[T.county_fips == "183"]
D = S7.miles(wake.lat.to_numpy(), wake.lon.to_numpy(), her.lat.to_numpy(), her.lon.to_numpy())
sets = {"wake_all": wake, "nearest_store_le_1mi": wake[D.min(1) <= 1], "nearest_le_2mi": wake[D.min(1) <= 2],
        "nearest_le_3mi": wake[D.min(1) <= 3], "store_tracts_only": wake[wake.GEOID.isin(her.tract_geoid.astype(str))]}
rows = []
for sname, tr in sets.items():
    for zname, zcol in (("pooled_z", "{d}_pooled_z"), ("raw_z", "{d}_raw_z")):
        for iu in (1_000.0, 10_000.0):
            _, regs = S7.run(tr, her, zcol, income_unit=iu)
            err_r2 = np.mean([abs(r.r2 - TARGET[r.outcome][3]) for r in regs.itertuples()])
            sign = np.mean([np.sign(getattr(r, k)) == np.sign(TARGET[r.outcome][i]) for r in regs.itertuples()
                            for i, k in enumerate(["b_pct_black_c", "b_income_c", "b_black_x_income"])])
            rows.append({"tracts": sname, "n_tracts": len(tr), "z": zname, "income_unit": iu, "mean_abs_r2_error": round(err_r2, 3),
                         "sign_agreement": round(float(sign), 2), "regs": regs.round(3).to_dict("records")})
res = sorted(rows, key=lambda r: (r["mean_abs_r2_error"], -r["sign_agreement"]))
for r in res[:8]:
    print(r["tracts"], r["n_tracts"], r["z"], int(r["income_unit"]), r["mean_abs_r2_error"], r["sign_agreement"])
best = res[0]
print("best:")
for g in best["regs"]:
    t = TARGET[g["outcome"]]
    print(f'{g["outcome"]:18s} ours b_black={g["b_pct_black_c"]:7.3f} b_inc={g["b_income_c"]:6.3f} b_x={g["b_black_x_income"]:6.3f} r2={g["r2"]:.3f} | hers {t}')
(OUT / "d9_table53.json").write_text(json.dumps(res, indent=1, default=float))
