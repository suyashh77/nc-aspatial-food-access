"""B4. Better dimension definitions (Pipeline B, switch 4).

    ../NC\\ Pipeline/.venv/bin/python b4_definitions.py

Problem (D1): Teresa's first-person text finds Availability poorly (recall ~0.2) and files "selection and variety" under
Acceptability, and "layout" under Accessibility, while the codebook every rater uses puts selection under Availability
and layout / checkout / pickup under Accommodation.

Variants of the 5 definition vectors:
  first_person            Teresa's text (Pipeline A)
  first_person_sent       the same text, each sentence embedded and averaged (D1's cheap gain)
  b4_text / b4_text_sent  new first-person text written from the raters' codebook (below)
  prototype               mean vector of the TRAINING half's gold reviews labelled with each dimension (data-defined)
  b4_text+prototype       average of the two
Each is scored on the B2 sentence route (a review's dimensions = union over its sentences' topics; topic centroid vs the
definitions; Teresa's softmax tau / extra-label delta, both re-tuned on the training half from a small grid).
20 random 50/50 splits of the gold set; we report the mean test score. Text variants don't learn from the training half
except tau / delta, so their test scores are close to their full-sample scores.

CAVEAT: until people label, the gold set is Claude's stand-in, coded with the same codebook the new text follows. Rerun
after step5_import_web_labels.py has written the human consensus (decisions/common.gold() picks it up automatically).

Outputs: out/b4_definitions.json, out/b4_definition_vectors.npz (the chosen variant, for Pipeline B)
"""
from itertools import product

from common_b import *
from common import gold, review_rows, score_sets
from definitions import FIRST_PERSON, sentences

B4_TEXT = {
    "Availability": "This store has the food I need. The shelves are full and well stocked, and items are rarely out of stock. "
    "There is a big selection and variety of products, brands and sizes, so I can find everything on my list in one trip. "
    "I wish there were more stores like this near me.",
    "Accessibility": "This store is easy to get to. It is close to my home and on my way, with an easy drive in and out. "
    "The parking lot has plenty of spaces and is easy to get around. I can get there by bus, bike or on foot, and the entrance "
    "and carts work for people with disabilities.",
    "Accommodation": "This store fits my schedule and my needs. It is open early and late, and it doesn't close early. Checkout is "
    "quick: enough registers are open, the lines are short, and self checkout works. The layout makes things easy to find. "
    "I can order online for pickup or delivery. They accept EBT, SNAP, WIC, cards and Apple Pay, and they carry options for my "
    "diet such as gluten free, halal or kosher.",
    "Affordability": "The prices here are low and I get good value for my money. There are weekly sales, coupons, deals and "
    "discounts with the loyalty card. It is cheaper than other stores, and I am not overcharged at the register.",
    "Acceptability": "This store meets my standards. The food is fresh and high quality, the produce and meat are never spoiled, "
    "and the store is clean and smells good. The staff are friendly, polite and helpful, and I feel safe and respected here. "
    "The store carries foods from my culture.",
}
TAUS, DELTAS = [0.05, 0.08, 0.11, 0.15], [0.10, 0.17, 0.25]


def embed(m, d, mode):
    if mode == "paragraph":
        return unit(m.encode([d[k] for k in DIMS], normalize_embeddings=True))
    return unit(np.stack([m.encode(ss, normalize_embeddings=True).mean(0) for ss in (sentences(d)[k] for k in DIMS)]))


def main():
    rng = np.random.default_rng(5)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(C.EMBEDDING_MODEL, device="cuda")
    g = gold()
    g = g[g.dims.map(len) > 0].reset_index(drop=True)
    g["row"] = review_rows(g.id)
    emb = np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")
    Eg = unit(np.asarray(emb[g.row.to_numpy()]))

    # B2 topic centroids (cached) and the topics of each gold review's sentences
    cpath = OUT / "b2_centroids.npz"
    srow = np.load(OUT / "b2_sentence_rows.npy")
    stop = np.load(OUT / "b2_sentence_topics.npy")
    if cpath.exists():
        z = np.load(cpath)
        cid, cen = z["ids"], z["cen"]
    else:
        Es = np.load(OUT / "b2_sentence_embeddings.npy", mmap_mode="r")
        ids = np.array(sorted(t for t in np.unique(stop) if t >= 0))
        pos = {t: i for i, t in enumerate(ids)}
        acc = np.zeros((len(ids), Es.shape[1]))
        for s in range(0, len(Es), 250_000):
            lab = stop[s:s + 250_000]
            k = lab >= 0
            np.add.at(acc, np.array([pos[t] for t in lab[k]]), np.asarray(Es[s:s + 250_000])[k])
        cid, cen = ids, unit(acc)
        np.savez(cpath, ids=cid, cen=cen)
    tab = pd.read_csv(OUT / "b2_topics.csv", index_col=0)
    gen = set(tab.index[tab.generic_auto])
    by_row = pd.Series(stop).groupby(srow).apply(lambda s: sorted(set(s) - {-1} - gen))
    g_topics = [by_row.get(r, []) for r in g.row]
    cpos = {t: i for i, t in enumerate(cid)}

    def predict(D, tau, delta):
        p = S4.sims_to_probs(pd.DataFrame(cen @ D.T, index=cid, columns=DIMS), tau, delta)
        lab = {t: {x for x in (r.pred_dim, r.pred_dim2, r.pred_dim3) if isinstance(x, str)} for t, r in p.iterrows()}
        return [set().union(*[lab[t] for t in ts]) if ts else set() for ts in g_topics], p.pred_dim.value_counts().to_dict()

    def per_dim_recall(pred, gold_sets):
        return {d: round(float(np.mean([d in p for p, gs in zip(pred, gold_sets) if d in gs])), 3) if any(d in gs for gs in gold_sets) else None for d in DIMS}

    fixed = {"first_person": embed(m, FIRST_PERSON, "paragraph"), "first_person_sent": embed(m, FIRST_PERSON, "sentences"),
             "b4_text": embed(m, B4_TEXT, "paragraph"), "b4_text_sent": embed(m, B4_TEXT, "sentences")}
    res = {k: [] for k in [*fixed, "prototype", "b4_text+prototype"]}
    detail = {k: [] for k in res}
    for split in range(20):
        perm = rng.permutation(len(g))
        tr, te = np.sort(perm[: len(g) // 2]), np.sort(perm[len(g) // 2:])
        proto = unit(np.stack([Eg[tr][[d in s for s in g.dims.iloc[tr]]].mean(0) for d in DIMS]))
        variants = dict(fixed, prototype=proto, **{"b4_text+prototype": unit(fixed["b4_text_sent"] + proto)})
        for name, D in variants.items():
            preds = {td: predict(D, *td) for td in product(TAUS, DELTAS)}       # once per setting
            best = max(preds, key=lambda td: score_sets([preds[td][0][i] for i in tr], list(g.dims.iloc[tr]))["f1"])
            pred, counts = preds[best]
            sc = score_sets([pred[i] for i in te], list(g.dims.iloc[te]))
            res[name].append(sc)
            detail[name].append({"tau_delta": best, "recall": per_dim_recall([pred[i] for i in te], list(g.dims.iloc[te])), "topics_by_dim": counts})
    R = {"gold_source": g.attrs.get("source"), "reviews": int(len(g)), "splits": 20, "variants": {}}
    for name in res:
        df = pd.DataFrame(res[name])
        rec = pd.DataFrame([x["recall"] for x in detail[name]]).mean().round(3).to_dict()
        R["variants"][name] = {"test_precision": round(df.precision.mean(), 3), "test_recall": round(df.recall.mean(), 3),
                               "test_f1": round(df.f1.mean(), 3), "test_f1_sd": round(df.f1.std(), 3), "recall_by_dim": rec,
                               "tau_delta_chosen": pd.Series([tuple(x["tau_delta"]) for x in detail[name]]).mode().iat[0],
                               "topics_by_dim_last_split": detail[name][-1]["topics_by_dim"]}
        print(name, R["variants"][name], flush=True)
    text_only = {k: v for k, v in R["variants"].items() if k in fixed}
    choice = max(text_only, key=lambda k: text_only[k]["test_f1"])
    R["chosen_text_variant"] = choice
    tau, delta = R["variants"][choice]["tau_delta_chosen"]
    np.savez(OUT / "b4_definition_vectors.npz", D=fixed[choice], tau=tau, delta=delta, name=choice)
    save("b4_definitions", R)


if __name__ == "__main__":
    main()
