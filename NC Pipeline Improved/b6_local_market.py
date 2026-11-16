"""B6 step 3: local markets. Score stores against what a person can actually reach, and score tracts by the stores
within a 16-minute drive (32 minutes as the sensitivity case), weighted by where people shop (Advan foot traffic, from Dewey) or, with
--weights equal, with every reachable store counting the same. Foot traffic is optional: Teresa's method doesn't use it.

    ../NC\\ Pipeline/.venv/bin/python b6_local_market.py --scores a     # Pipeline A's pooled store scores (isolates B6)
    ../NC\\ Pipeline/.venv/bin/python b6_local_market.py --scores b5    # B5's pooled scores
    ... --weights equal                                                    # no foot-traffic data (outputs get _equal)

Problems with Pipeline A's stage 7:
  - Every store is z-scored against the whole state, so a mountain-town store is compared with Charlotte's.
  - Tract scores weight stores by exp(-miles / 1 mile), an urban rule: in a rural tract the score is effectively the
    single nearest store, however far.
What B6 does, for each tract i (the "person", at its internal point) and market M_i = stores within T minutes by car
(b6_drive_times.py, OSM roads):
  Tract access     E_ij = sum_{l in M_i} v_l s_lj / sum v_l      v = visits per week (Advan, latest 12 months; stores Advan
                                                                     doesn't see get their category's median), or v = 1
                   best_ij = max_{l in M_i} s_lj, n_i = number of stores in M_i
                   each then put on a statewide scale (population-weighted z across tracts).
  Store local z    z_lj(i) = (s_lj - mean_{M_i} s) / sd_{M_i} s    (markets with >= 3 scored stores)
                   store's local z = population-weighted mean over the tracts that can reach it: how the store compares
                   with the other options of the people who can get to it.
Regression: the stage-7 model (population-weighted; % Black share, income $10k, interaction, centred) on each outcome,
next to A's stage-7 results, plus drive minutes to the nearest store in place of straight-line miles.

Outputs: out/b6_tracts_<scores>.csv, out/b6_store_local_z_<scores>.csv, out/b6_local_market_<scores>.json
"""
import argparse

from common_b import *
from step7_tracts import nc_tracts, wls

ap = argparse.ArgumentParser()
ap.add_argument("--scores", choices=["a", "b5", "b"], default="a")
ap.add_argument("--weights", choices=["visits", "equal"], default="visits",
                help="visits = Advan foot traffic (Dewey; optional data, not in Teresa's method); equal = every reachable store counts the same")
args = ap.parse_args()

if args.scores == "a":
    sc = pd.read_csv(C.DATA / "step6_store_scores.csv").set_index("store_id")
    S = pd.DataFrame({d: sc[f"{d}_pooled_mean_signal"] for d in DIMS})
elif args.scores == "b5":
    sc = pd.read_csv(OUT / "b5_store_scores.csv").set_index("store_id")
    S = pd.DataFrame({d: sc[f"{d}_b5_mean"] for d in DIMS})
else:                                   # full Pipeline B (b_pipeline.py)
    sc = pd.read_csv(OUT / "b_store_scores.csv").set_index("store_id")
    S = pd.DataFrame({d: sc[f"{d}_b_mean"] for d in DIMS})

dt = pd.read_parquet(OUT / "b6_drive_times.parquet")
if args.weights == "visits":            # optional: Dewey/Advan foot traffic
    ft = pd.read_csv(C.ROOT / "grocery-foot-traffic/stores.csv", usecols=["master_store_id", "category", "visits_per_week_latest12"])
    ft = ft.rename(columns={"master_store_id": "store_id"}).set_index("store_id")
    med = ft.groupby("category").visits_per_week_latest12.median()
    v = ft.visits_per_week_latest12.fillna(ft.category.map(med)).fillna(ft.visits_per_week_latest12.median())
    dt["v"] = dt.store_id.map(v).fillna(v.median()).to_numpy()
else:                                   # every reachable store counts equally, as in Teresa's method
    dt["v"] = 1.0
SFX = "" if args.weights == "visits" else "_equal"
tr = nc_tracts().set_index("GEOID")
R = {"scores": args.scores, "store_weights": args.weights, "tracts": int(len(tr))}


def wz(x, w):
    ok = x.notna() & w.notna() & (w > 0)
    m = np.average(x[ok], weights=w[ok])
    s = np.sqrt(np.average((x[ok] - m) ** 2, weights=w[ok]))
    return (x - m) / s


out = tr.copy()
nearest = dt.groupby("GEOID").minutes.min()
out["drive_min_nearest"] = nearest
out["drive_min_nearest_z"] = wz(out.drive_min_nearest, out["pop"])
local = {}
for T in (16, 32):
    P = dt[dt.minutes <= T]
    out[f"n_stores_{T}"] = P.groupby("GEOID").size().reindex(out.index, fill_value=0)
    R[f"tracts_no_store_{T}min"] = int((out[f"n_stores_{T}"] == 0).sum())
    R[f"people_no_store_{T}min"] = int(out["pop"][out[f"n_stores_{T}"] == 0].sum())
    for d in DIMS:
        q = P.assign(s=P.store_id.map(S[d])).dropna(subset=["s"])
        g = q.assign(vs=q.v * q.s).groupby("GEOID")
        E = (g.vs.sum() / g.v.sum()).reindex(out.index)
        out[f"E_{d}_{T}"] = wz(E, out["pop"])
        out[f"best_{d}_{T}"] = wz(g.s.max().reindex(out.index), out["pop"])
        if T == 16:
            # store local z, person-centred
            mk = q.groupby("GEOID").apply(lambda x: pd.Series({"m": np.average(x.s, weights=x.v),
                                                                 "sd": np.sqrt(np.average((x.s - np.average(x.s, weights=x.v)) ** 2, weights=x.v)),
                                                                 "k": len(x)}), include_groups=False)
            mk = mk[(mk.k >= 3) & (mk.sd > 1e-9)]
            z = q.join(mk, on="GEOID", how="inner")
            z["z"] = (z.s - z.m) / z.sd
            z["pop"] = z.GEOID.map(tr["pop"]).fillna(0)
            z = z[z["pop"] > 0]
            local[d] = z.groupby("store_id").apply(lambda x: np.average(x.z, weights=x["pop"]), include_groups=False)
    # fallback: no store within 16 -> use the 32-minute market (rural)
for d in DIMS:
    out[f"E_{d}_fallback"] = out[f"E_{d}_16"].fillna(out[f"E_{d}_32"])

lz = pd.DataFrame(local)
lz.index.name = "store_id"
state_z = (S - S.mean()) / S.std()
R["store_local_vs_state_z_spearman"] = {d: round(float(lz[d].corr(state_z[d].reindex(lz.index), method="spearman")), 3) for d in DIMS}
R["stores_with_local_z"] = int(lz.notna().any(axis=1).sum())

# regressions
regs = []
for y in ["drive_min_nearest_z"] + [f"E_{d}_{s}" for d in DIMS for s in ("16", "32", "fallback")] + [f"best_{d}_16" for d in DIMS]:
    r = wls(out.reset_index(), y)
    regs.append(r)
regs = pd.DataFrame(regs)
a7 = pd.read_csv(C.DATA / "step7_regressions_nc.csv")
R["regressions"] = regs.round(4).to_dict("records")
R["A_stage7"] = a7.round(4).to_dict("records")
out.reset_index().to_csv(OUT / f"b6_tracts_{args.scores}{SFX}.csv", index=False)
lz.reset_index().to_csv(OUT / f"b6_store_local_z_{args.scores}{SFX}.csv", index=False)
save(f"b6_local_market_{args.scores}{SFX}", R)
cols = ["outcome", "n_tracts", "b_pct_black_c", "p_pct_black_c", "b_income_c", "p_income_c", "b_black_x_income", "r2"]
print(regs[cols].round(3).to_string(index=False))
print(a7[cols].round(3).to_string(index=False))
print(json.dumps({k: v for k, v in R.items() if k not in ("regressions", "A_stage7")}, indent=1))
