"""Pipeline B end to end: every adopted switch on, scored against Pipeline A.

    ../NC\\ Pipeline/.venv/bin/python b_pipeline.py

  Which dimension(s)   B2 sentence topics; topic centroid vs the B4 definitions (out/b4_definition_vectors.npz, with its
                       tau / delta); generic topics set aside (out/b2_manual_scores.csv from a human check of the sentence
                       topics replaces the automatic list when it exists). A review counts toward the union of its
                       sentences' dimensions. (B1, mood removal, is not used here: B2's sentence topics already put rude
                       and friendly cashiers together, overlap 0.90.)
  Good or bad          B7: the mean text sentiment of the review's sentences on that dimension (-1..+1).
                       Variant "stars": Teresa's star signal instead, so B7's own effect can be read off.
  Store score          B5 pooling. Text signal (continuous): ybar_j ~ N(theta_j, sigma_w / sqrt(n_j)); stars (counts):
                       ordered-logit multinomial. Both: theta_j = county + category + tau * eta_j, non-centred.
  Neighbourhood        B6: run b6_local_market.py --scores b afterwards.

Outputs: out/b_store_scores.csv ({dim}_b_mean, _lo95, _hi95, _z, _n; {dim}_bstars_mean), out/b_pipeline.json
"""
import time

import pymc as pm

from common_b import *
from step6_store_scores import stores_table
from b_review_dims import review_dims
import b5_pooling as B5

t0 = time.time()
R = {}
rv = pd.read_parquet(C.DATA / "step1_reviews.parquet", columns=["store_id", "rating"])
rd, info = review_dims()           # b_review_dims.py: sentence topics -> B4 definitions -> review x dimension, B7 text sentiment
R.update(info)
rd.to_parquet(OUT / "b_review_dims.parquet", index=False)   # read by ../NC Pipeline/export_review_table.py
rd["store_id"] = rv.store_id.to_numpy()[rd.row]
rd["stars"] = rv.rating.to_numpy()[rd.row]
rd = rd.dropna(subset=["stars"])
rd["star_signal"] = rd.stars.map({1: -1, 2: 0, 3: 0, 4: 0, 5: 1})
R["review_dim_pairs"] = int(len(rd))
R["reviews_with_a_dim"] = int(rd.row.nunique())
R["share_reviews_by_dim"] = (rd.groupby("dim").row.nunique() / len(rv)).round(3).to_dict()

s = rd.groupby(["store_id", "dim"]).agg(n_reviews=("row", "size"), text_mean=("text", "mean"), text_var=("text", "var"),
                                         n_low=("star_signal", lambda x: (x == -1).sum()), n_neutral=("star_signal", lambda x: (x == 0).sum()),
                                         n_high=("star_signal", lambda x: (x == 1).sum())).reset_index()
st = stores_table().set_index("store_id")
county = st.county.fillna("UNKNOWN")
cat = st.category.fillna("other")
cat = cat.where(cat.map(cat.value_counts()) >= 30, "other")


def fit_text(df):
    sw = float(np.sqrt(np.nansum((df.n_reviews - 1) * df.text_var.fillna(0)) / max((df.n_reviews - 1).sum(), 1)))
    ci, cl = pd.factorize(county.reindex(df.index).fillna("UNKNOWN"))
    ki, kl = pd.factorize(cat.reindex(df.index).fillna("other"))
    with pm.Model():
        mu = pm.Normal("mu", 0, 0.5)
        sd_c, sd_k, tau_ = pm.HalfNormal("sd_county", 0.3), pm.HalfNormal("sd_category", 0.3), pm.HalfNormal("tau", 0.5)
        zc, zk = pm.ZeroSumNormal("z_county", sigma=1.0, shape=len(cl)), pm.ZeroSumNormal("z_category", sigma=1.0, shape=len(kl))  # sum to zero: mu is the level
        eta = pm.Normal("eta", 0, 1, shape=len(df))
        theta = pm.Deterministic("theta", mu + sd_c * zc[ci] + sd_k * zk[ki] + tau_ * eta)
        pm.Normal("obs", mu=theta, sigma=sw / np.sqrt(df.n_reviews.to_numpy(float)), observed=df.text_mean.to_numpy(float))
        idata = pm.sample(**B5.DRAWS)
    return idata.posterior["theta"].stack(s=("chain", "draw")).values, idata, sw


a6 = pd.read_csv(C.DATA / "step6_store_scores.csv").set_index("store_id")
wide = pd.DataFrame(index=sorted(s.store_id.unique()))
wide.index.name = "store_id"
R["diagnostics"], R["vs_A"] = {}, {}
for d in DIMS:
    df = s[s.dim == d].set_index("store_id")
    th, idata, sw = fit_text(df)
    wide[f"{d}_n"] = df.n_reviews
    wide[f"{d}_b_mean"] = pd.Series(th.mean(1), index=df.index)
    wide[f"{d}_b_lo95"] = pd.Series(np.quantile(th, .025, 1), index=df.index)
    wide[f"{d}_b_hi95"] = pd.Series(np.quantile(th, .975, 1), index=df.index)
    wide[f"{d}_b_z"] = (wide[f"{d}_b_mean"] - wide[f"{d}_b_mean"].mean()) / wide[f"{d}_b_mean"].std()
    R["diagnostics"][f"text_{d}"] = {**B5.diag(idata), "sigma_w": round(sw, 3)}
    th2, idata2 = B5.fit_B5(df, county, cat)
    wide[f"{d}_bstars_mean"] = pd.Series(th2.mean(1), index=df.index)
    R["diagnostics"][f"stars_{d}"] = B5.diag(idata2)
    Am = a6[f"{d}_pooled_mean_signal"]
    j = pd.concat([wide[f"{d}_b_mean"], wide[f"{d}_bstars_mean"], Am], axis=1, keys=["b", "bs", "a"]).dropna()
    R["vs_A"][d] = {"stores_B": int(df.shape[0]), "stores_A": int(Am.notna().sum()), "stores_both": int(len(j)),
                    "spearman_B_vs_A": round(float(j.b.corr(j.a, method="spearman")), 3),
                    "spearman_Bstars_vs_A": round(float(j.bs.corr(j.a, method="spearman")), 3),
                    "spearman_B_vs_Bstars": round(float(j.b.corr(j.bs, method="spearman")), 3)}
    print(d, R["diagnostics"][f"text_{d}"], R["vs_A"][d], flush=True)
wide.reset_index().to_csv(OUT / "b_store_scores.csv", index=False)
R["seconds"] = round(time.time() - t0)
save("b_pipeline", R)
print(json.dumps(R, indent=1, default=float)[:3000])
