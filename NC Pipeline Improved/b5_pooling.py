"""B5. Better pooling of store scores (Pipeline B, switch 5).

    ../NC\\ Pipeline/.venv/bin/python b5_pooling.py               # holdout test + full NC fit
    ../NC\\ Pipeline/.venv/bin/python b5_pooling.py --skip-holdout

Problems with Teresa's model (Pipeline A, stage 6):
  1. SE floor. SE comes from each store's own reviews; when they all agree (all 5 stars) SE = 0, clipped to 1e-6, so the
     model treats a 3-review store as perfectly measured and never shrinks it.
  2. One mean for every store: a dollar store is pulled toward the same average as a supermarket.
  3. Centred model: with few, similar stores (one small county, frozen arm, Accessibility on 21 stores) it fails to converge.

Three models, one dimension at a time, on the same store x dimension counts as stage 6:
  A    Teresa:  ybar_j ~ N(theta_j, se_j),  se_j from the store's own reviews (floor 1e-6), theta_j ~ N(mu, tau).
  B5a  SE fix:  same, but se_j = sigma_w / sqrt(n_j) with sigma_w the within-store SD pooled over all stores.
                Non-centred.
  B5   Ordinal: the store's counts of low (1 star) / neutral (2-4) / high (5) reviews ~ Multinomial(ordered logit(theta_j)),
                theta_j = county effect + category effect + tau * eta_j, all non-centred;
                county and category effects sum to zero, so they can't trade off with the cutpoints (the overall level). Score = P(high) - P(low),
                on the same -1..+1 scale as Teresa's mean signal.

Test: each store's reviews are split at random into two halves. Each model is fitted on half 1 and predicts the raw
mean signal of half 2 (stores with >= 3 reviews in half 2). Error = n-weighted mean squared error; lower is better.
Then the chosen model is fitted on all reviews.

Outputs: out/b5_store_scores.csv (store_id + per dim: b5_mean, b5_lo95, b5_hi95, b5_z, n), out/b5_pooling.json
"""
import argparse
import time

import pymc as pm
import pytensor.tensor as pt

from common_b import *
from step6_store_scores import aggregate, stores_table, validity

rng = np.random.default_rng(7)
DRAWS = dict(draws=1000, tune=2000, chains=4, target_accept=0.95, random_seed=11, nuts_sampler="nutpie", progressbar=False)


def reviews():
    valid, vsrc = validity("nc")
    rd = pd.read_parquet(C.DATA / "step4_review_dims.parquet")
    rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["id", "rating"])
    rd["rating"] = rv.rating.to_numpy()
    rd = rd[rd.topic.isin(valid) & rd.rating.notna()]
    return rd, vsrc


def fit_A(df):
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 0.5)
        tau = pm.HalfNormal("tau", 0.5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=len(df))
        pm.Normal("obs", mu=theta, sigma=df.raw_se_signal.to_numpy(float), observed=df.raw_mean_signal.to_numpy(float))
        idata = pm.sample(**DRAWS)
    return idata.posterior["theta"].stack(s=("chain", "draw")).values, idata


def fit_B5a(df, sigma_w):
    se = sigma_w / np.sqrt(df.n_reviews.to_numpy(float))
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 0.5)
        tau = pm.HalfNormal("tau", 0.5)
        eta = pm.Normal("eta", 0, 1, shape=len(df))
        theta = pm.Deterministic("theta", mu + tau * eta)
        pm.Normal("obs", mu=theta, sigma=se, observed=df.raw_mean_signal.to_numpy(float))
        idata = pm.sample(**DRAWS)
    return idata.posterior["theta"].stack(s=("chain", "draw")).values, idata


def fit_B5(df, county, category):
    ci, cl = pd.factorize(county.reindex(df.index).fillna("UNKNOWN"))
    ki, kl = pd.factorize(category.reindex(df.index).fillna("other"))
    counts = df[["n_low", "n_neutral", "n_high"]].to_numpy(int)
    with pm.Model():
        cut = pm.Normal("cut", 0, 2, shape=2, transform=pm.distributions.transforms.ordered, initval=np.array([-1.5, 0.5]))
        sd_c = pm.HalfNormal("sd_county", 0.5)
        sd_k = pm.HalfNormal("sd_category", 0.5)
        tau = pm.HalfNormal("tau", 1.0)
        zc = pm.ZeroSumNormal("z_county", sigma=1.0, shape=len(cl))       # sum to zero: the cutpoints carry the level,
        zk = pm.ZeroSumNormal("z_category", sigma=1.0, shape=len(kl))     # so level and group effects can't trade off
        eta = pm.Normal("eta", 0, 1, shape=len(df))
        theta = sd_c * zc[ci] + sd_k * zk[ki] + tau * eta
        p_low = pm.math.sigmoid(cut[0] - theta)
        p_le = pm.math.sigmoid(cut[1] - theta)
        score = pm.Deterministic("theta", (1 - p_le) - p_low)        # expected signal, -1..+1
        probs = pt.stack([p_low, p_le - p_low, 1 - p_le], axis=1)
        pm.Multinomial("y", n=counts.sum(1), p=probs, observed=counts)
        idata = pm.sample(**DRAWS)
    return idata.posterior["theta"].stack(s=("chain", "draw")).values, idata


def diag(idata):
    import arviz as az
    v = [k for k in ("mu", "tau", "cut", "sd_county", "sd_category") if k in idata.posterior]
    rh = max(float(az.rhat(idata, var_names=[k])[k].max()) for k in v) if v else float("nan")
    return {"rhat_max_hyper": round(rh, 3), "rhat_max_theta": round(float(az.rhat(idata, var_names=["theta"])["theta"].max()), 3),
            "ess_min_theta": int(az.ess(idata, var_names=["theta"])["theta"].min()), "divergences": int(idata.sample_stats["diverging"].sum())}


def sigma_within(s):
    """Pooled within-store SD of the signal: sum over stores of n * var_j / sum n, var_j = p_low + p_high - mean^2."""
    v = (s.n_low + s.n_high) / s.n_reviews - s.raw_mean_signal ** 2
    return float(np.sqrt((s.n_reviews * v).sum() / s.n_reviews.sum()))


def fit_all(s, county, category, which=("A", "B5a", "B5")):
    out, dg = {}, {}
    for dim in DIMS:
        df = s[s.dim == dim].set_index("store_id")
        sw = sigma_within(df)
        for m in which:
            th, idata = fit_A(df) if m == "A" else fit_B5a(df, sw) if m == "B5a" else fit_B5(df, county, category)
            out[(m, dim)] = pd.DataFrame({"mean": th.mean(1), "lo95": np.quantile(th, .025, 1), "hi95": np.quantile(th, .975, 1)}, index=df.index)
            dg[f"{m}_{dim}"] = diag(idata)
            print(m, dim, dg[f"{m}_{dim}"], flush=True)
    return out, dg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-holdout", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rd, vsrc = reviews()
    st = stores_table().set_index("store_id")
    county = st.county.fillna("UNKNOWN")
    cat = st.category.fillna("other")
    cat = cat.where(cat.map(cat.value_counts()) >= 30, "other")
    R = {"validity": vsrc, "reviews": int(len(rd))}

    if not args.skip_holdout:
        half = rng.random(len(rd)) < 0.5
        _, s1 = aggregate(rd[half])
        _, s2 = aggregate(rd[~half])
        fits, dg = fit_all(s1, county, cat)
        R["holdout_diagnostics"] = dg
        res = {}
        for dim in DIMS:
            test = s2[(s2.dim == dim) & (s2.n_reviews >= 3)].set_index("store_id")
            train_n = s1[s1.dim == dim].set_index("store_id").n_reviews
            row = {}
            for m in ("raw", "A", "B5a", "B5"):
                if m == "raw":
                    pred = s1[s1.dim == dim].set_index("store_id").raw_mean_signal
                else:
                    pred = fits[(m, dim)]["mean"]
                j = test.join(pred.rename("pred"), how="inner").join(train_n.rename("ntrain"))
                err = (j.pred - j.raw_mean_signal) ** 2
                row[m] = round(float(np.average(err, weights=j.n_reviews)), 5)
                small = j.ntrain <= 5
                row[m + "_small_train"] = round(float(np.average(err[small], weights=j.n_reviews[small])), 5) if small.any() else None
                row["stores"] = int(len(j))
                row["stores_small_train"] = int(small.sum())
            res[dim] = row
            print(dim, row, flush=True)
        R["holdout_mse"] = res
        (OUT / "b5_holdout.json").write_text(json.dumps({"holdout_mse": res, "holdout_diagnostics": dg}, indent=1))
    elif (OUT / "b5_holdout.json").exists():
        R.update(json.loads((OUT / "b5_holdout.json").read_text()))

    long, s = aggregate(rd)
    fits, dg = fit_all(s, county, cat)
    R["full_diagnostics"] = dg
    a6 = pd.read_csv(C.DATA / "step6_store_scores.csv").set_index("store_id")
    wide = pd.DataFrame(index=sorted(s.store_id.unique()))
    cmp_ = {}
    for dim in DIMS:
        df = s[s.dim == dim].set_index("store_id")
        b = fits[("B5", dim)]
        wide[f"{dim}_n"] = df.n_reviews
        wide[f"{dim}_raw"] = df.raw_mean_signal
        for k in ("mean", "lo95", "hi95"):
            wide[f"{dim}_b5_{k}"] = b[k]
        wide[f"{dim}_b5a_mean"] = fits[("B5a", dim)]["mean"]
        m_ = wide[f"{dim}_b5_mean"]
        wide[f"{dim}_b5_z"] = (m_ - m_.mean()) / m_.std()
        m_ = m_.reindex(df.index)
        Am = fits[("A", dim)]["mean"].reindex(df.index)
        floor = df.raw_se_signal <= 1e-6
        small = df.n_reviews <= 5
        cmp_[dim] = {
            "spearman_B5_vs_A": round(float(m_.corr(Am, method="spearman")), 3),
            "spearman_B5a_vs_A": round(float(fits[("B5a", dim)]["mean"].corr(Am, method="spearman")), 3),
            "stores_at_se_floor_in_A": int(floor.sum()),
            "mean_abs_shrink_small_n_A": round(float((Am - df.raw_mean_signal)[small].abs().mean()), 3),
            "mean_abs_shrink_small_n_B5": round(float((m_ - df.raw_mean_signal)[small].abs().mean()), 3),
            "top10_A_median_n": float(df.n_reviews.loc[Am.nlargest(10).index].median()),
            "top10_B5_median_n": float(df.n_reviews.loc[m_.nlargest(10).index].median()),
        }
        print(dim, cmp_[dim], flush=True)
    R["vs_A"] = cmp_
    wide.index.name = "store_id"
    wide.reset_index().to_csv(OUT / "b5_store_scores.csv", index=False)
    R["seconds"] = round(time.time() - t0)
    save("b5_pooling", R)
    print(json.dumps(R, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
