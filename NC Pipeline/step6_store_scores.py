"""Step 6: store x dimension scores with Bayesian partial pooling (Teresa notebook 6), Pipeline A = her method.

    .venv/bin/python step6_store_scores.py                 # all NC stores
    .venv/bin/python step6_store_scores.py --tag raleigh --stores raleigh

As in her notebook 6:
  1. Keep reviews whose topic was judged valid (human check; provisional list until it's done).
  2. Signal from stars: 1 -> -1, 2-4 -> 0, 5 -> +1. A review counts toward each of its topic's labels
     (pred_dim, pred_dim2, pred_dim3).
  3. Store x dimension: mean signal, n, SE = sqrt((p_low + p_high - mean^2) / n), clipped at 1e-6.
  4. One model per dimension:  ybar_j ~ N(theta_j, se_j),  theta_j ~ N(mu, tau),  mu ~ N(0, 0.5),  tau ~ HalfN(0.5).
     2,000 draws, 2,000 tuning, 4 chains, target_accept 0.95, seed 123. Sampled with nutpie (same model, faster
     than PyMC's default sampler; the posterior is the same).
  5. z = (pooled mean - mean of pooled means) / SD across stores.
Only change: stores are labelled with the store name (her code used the reviewer's name, a labelling bug).

Outputs (data/): step6_store_scores[_tag].csv, step6_store_dim_long[_tag].csv, step6_pooling_summary[_tag].csv, step6_meta[_tag].json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time

import numpy as np
import pandas as pd

import config as C
from definitions import DIMS

LABELS = ["pred_dim", "pred_dim2", "pred_dim3"]


def validity(arm: str = "nc") -> tuple[set, str]:
    if arm == "frozen":
        inv = set(pd.read_csv(C.DATA / "step4_frozen_invalid_topics.csv").Topic)
        return set(range(200)) - inv, "Teresa's own validity (her 18 invalid Raleigh topics)"
    f = C.DATA / "step5_manual_scores.csv"
    if f.exists():
        s = pd.read_csv(f)
        return set(s.Topic[s.manual_topic_status == "valid"]), "human check (step5_manual_scores.csv)"
    p = pd.read_csv(C.DATA / "parents_draft.csv")
    bad = p.parent.str.startswith(("General praise", "No clear subject"))
    return set(p.topic[~bad]), "provisional (parents_draft.csv: general praise / no clear subject = invalid)"


def stores_table() -> pd.DataFrame:
    from export_teresa_format import FRAME_DB, county_of, load_polygons, read_only
    with read_only(C.REVIEWS_DB) as db:
        st = pd.read_sql("SELECT store_id, place_id, feature_id, store_name, outscraper_name, city, category, "
                         "source_lat AS lat, source_lon AS lon, rating, target_reviews FROM stores", db)
    with read_only(FRAME_DB) as fr:
        known = pd.read_sql("SELECT place_id, county AS frame_county FROM stores WHERE county <> '' AND place_id IS NOT NULL", fr)
    known = known.drop_duplicates("place_id")
    known["frame_county"] = known.frame_county.str.upper().str.replace(" ", "")
    st = st.merge(known, how="left", on="place_id")
    polys, boxes = load_polygons()
    st["county"] = [county_of(x, y, polys, boxes) for x, y in zip(st.lon, st.lat)]
    st["county"] = st.county.fillna(st.frame_county)
    st["name"] = st.outscraper_name.fillna(st.store_name)
    return st.drop(columns=["frame_county", "outscraper_name"])


def raleigh_store_ids(st: pd.DataFrame) -> set:
    th = pd.read_csv(C.ROOT / "Teresa Project/Raleigh_Data/raleigh_ring_reviews_with_dims.csv", low_memory=False,
                     usecols=["placeId", "fid"])
    th = th.drop_duplicates("placeId")
    return set(st.store_id[st.place_id.isin(th.placeId) | st.feature_id.isin(th.fid)])


def aggregate(rv: pd.DataFrame) -> pd.DataFrame:
    """Notebook 6 cells 6-12."""
    rv = rv.copy()
    rv["dim_signal"] = rv.rating.map(C.SIGNAL).astype(float)
    rv["dim_sentiment"] = np.select([rv.rating == 1, rv.rating.isin([2, 3, 4]), rv.rating == 5], ["Low", "Neutral", "High"], default="Neutral")
    long = rv.melt(id_vars=["id", "store_id", "rating", "dim_signal", "dim_sentiment"], value_vars=LABELS, var_name="label_rank", value_name="dim")
    long = long[long.dim.isin(DIMS)]
    s = long.groupby(["store_id", "dim"], as_index=False).agg(
        raw_mean_signal=("dim_signal", "mean"), n_reviews=("dim_signal", "size"),
        n_low=("dim_sentiment", lambda x: (x == "Low").sum()), n_neutral=("dim_sentiment", lambda x: (x == "Neutral").sum()),
        n_high=("dim_sentiment", lambda x: (x == "High").sum()))
    p_low, p_high = s.n_low / s.n_reviews, s.n_high / s.n_reviews
    s["raw_se_signal"] = np.sqrt((p_low + p_high - s.raw_mean_signal ** 2).clip(lower=0) / s.n_reviews).clip(lower=C.SE_FLOOR)
    return long, s


def pool_one(df: pd.DataFrame, dim: str):
    """Notebook 6 cell 27, same model and settings."""
    import pymc as pm
    ybar, se = df.raw_mean_signal.to_numpy(float), df.raw_se_signal.to_numpy(float)
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 0.5)
        tau = pm.HalfNormal("tau", 0.5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=len(df))
        pm.Normal("obs", mu=theta, sigma=se, observed=ybar)
        idata = pm.sample(draws=C.POOL_DRAWS, tune=C.POOL_TUNE, chains=C.POOL_CHAINS, target_accept=C.POOL_TARGET_ACCEPT,
                          random_seed=C.POOL_SEED, nuts_sampler="nutpie", progressbar=False)
    return idata


def main() -> None:
    import arviz as az
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", default="all", help="all | raleigh | <COUNTY NAME>")
    ap.add_argument("--tag", default="")
    ap.add_argument("--arm", default="nc", choices=["nc", "frozen"],
                    help="nc = NC topic model (step4_review_dims); frozen = Teresa's Raleigh model (step4_review_dims_frozen)")
    args = ap.parse_args()
    t0 = time.time()
    valid, vsrc = validity(args.arm)
    args.review_dims = "step4_review_dims.parquet" if args.arm == "nc" else "step4_review_dims_frozen.parquet"
    rd = pd.read_parquet(C.DATA / args.review_dims)
    rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id", "rating"])
    assert (rd.id.to_numpy() == rv.id.to_numpy()).all()
    rd["rating"] = rv.rating.to_numpy()
    st = stores_table()
    if args.stores == "raleigh":
        keep = raleigh_store_ids(st)
    elif args.stores != "all":
        keep = set(st.store_id[st.county == args.stores.upper().replace(" ", "")])
    else:
        keep = set(st.store_id)
    rd = rd[rd.store_id.isin(keep)]
    rd = rd[rd.topic.isin(valid)]
    long, s = aggregate(rd)
    summary, wide = [], st[st.store_id.isin(keep)].set_index("store_id")
    for dim in DIMS:
        df = s[(s.dim == dim) & (s.n_reviews > 0)].dropna(subset=["raw_mean_signal", "raw_se_signal"]).set_index("store_id")
        for c in ("raw_mean_signal", "n_reviews", "raw_se_signal"):
            wide[f"{dim}_{c}"] = df[c]
        if len(df) < 3:
            continue
        idata = pool_one(df, dim)
        th = idata.posterior["theta"].stack(sample=("chain", "draw")).values
        wide[f"{dim}_pooled_mean_signal"] = pd.Series(th.mean(1), index=df.index)
        wide[f"{dim}_pooled_lo95"] = pd.Series(np.quantile(th, 0.025, axis=1), index=df.index)
        wide[f"{dim}_pooled_hi95"] = pd.Series(np.quantile(th, 0.975, axis=1), index=df.index)
        wide[f"{dim}_shrinkage"] = wide[f"{dim}_pooled_mean_signal"] - wide[f"{dim}_raw_mean_signal"]
        pm_ = wide[f"{dim}_pooled_mean_signal"]
        wide[f"{dim}_pooled_z"] = (pm_ - pm_.mean()) / pm_.std()
        sm = az.summary(idata, var_names=["mu", "tau"])
        rhat_theta = float(az.rhat(idata, var_names=["theta"])["theta"].max())
        ess_theta = float(az.ess(idata, var_names=["theta"])["theta"].min())
        summary.append({"dim": dim, "n_stores": len(df), "mu_mean": sm.loc["mu", "mean"], "tau_mean": sm.loc["tau", "mean"],
                        "rhat_max_mu_tau": float(sm.r_hat.max()), "rhat_max_theta": rhat_theta, "ess_min_theta": ess_theta,
                        "divergences": int(idata.sample_stats["diverging"].sum())})
        print(summary[-1], flush=True)
    sfx = f"_{args.tag}" if args.tag else ""
    wide.reset_index().to_csv(C.DATA / f"step6_store_scores{sfx}.csv", index=False)
    s.to_csv(C.DATA / f"step6_store_dim_long{sfx}.csv", index=False)
    pd.DataFrame(summary).to_csv(C.DATA / f"step6_pooling_summary{sfx}.csv", index=False)
    meta = {"stores_filter": args.stores, "validity": vsrc, "valid_topics": len(valid), "reviews_in_valid_topics": int(len(rd)),
            "stores": int(len(keep)), "review_dims_file": args.review_dims, "seconds": round(time.time() - t0), "pooling": summary}
    (C.DATA / f"step6_meta{sfx}.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps({k: v for k, v in meta.items() if k != "pooling"}, indent=2))


if __name__ == "__main__":
    main()
