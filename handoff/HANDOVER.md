# Handover

Suyash Sapre, Nov 2026. For Dr. Rachunok, Caroline, and whoever picks up the code.

## The question

- Food access is usually measured by distance (food deserts). Distance misses price, quality, stock, hours, how people are treated.
- Teresa Groton's thesis used Google reviews as a large unplanned survey of what shopping at a store is like. She sorted reviews into the five A's (Penchansky & Thomas 1981, adapted to food by Caspi et al. 2012). 49 Raleigh stores, ~25K reviews.
- This project runs her method on every grocery store in NC, then improves it.

## What exists now

- 4,442 stores in all 100 counties. 1,941,538 reviews, 977,398 with text. Frozen 2026-10-24.
- Pipeline A (Teresa's method at NC scale) and Pipeline B (improvements) both run end to end.
- Store and tract scores from both pipelines, plus equity regressions.
- A labeling web page, live and empty.
- A one-command rerun (`run_after_labeling.sh`) for after labeling. Tested end to end on 2026-11-15 with five fake raters.

## How the data was built

1. **Store list.** Combined USDA SNAP, Overture Maps, OpenStreetMap and Dewey/SafeGraph + Data Axle. geomatch (github.com/suyashh77/geomatch) linked records by distance and name. 91,461 records → 36,820 food retailers → 16,751 candidates probed on Google Maps.
   - The spring version (Overture only) kept 14–38% of stores and was worst in rural eastern NC. That's why it was rebuilt.
2. **What counts as a grocery store.** ~300 Google categories ruled on by hand.
   - In: supermarkets, grocers, discount and dollar stores, ethnic grocers, butchers, seafood, co-ops. Any chain with 5+ NC locations.
   - Out: gas stations, convenience stores, farmers' markets, restaurants, closed stores.
   - Nothing deleted. Out-of-scope (4,736), closed (67), duplicates (221) each sit in their own database with a reason.
3. **Reviews.** Own Playwright scraper, opens each store by exact Google place ID. Ran early 2026 to Oct 24 on NC State Linux machines.
   - A Sept audit found the first version scraped the wrong store sometimes (it searched by name). Data and code were repaired before the rest was collected.
4. **Coverage.** 99.6% of reviews vs Google's counts capped at 1,500 per store. 87.9% vs true counts (big stores hold only their newest reviews). 29,321 reviews set aside with reasons, none deleted.

## The pipeline

Pipeline A, Teresa's method, changed only where NC's size forces it (code: `NC Pipeline/`):

- **Step 1.** Clean text, embed with all-MiniLM-L12-v2. Added 14 NC chain names to her strip list.
- **Step 2.** UMAP to 5 dimensions (GPU, cuML).
- **Step 3.** HDBSCAN + BERTopic. Her min cluster size of 120 gives thousands of topics at NC scale. Grid search picked 2,000 → 83 topics, same coherence as hers (0.529 vs 0.528).
- **Step 4.** Match each topic to the closest dimension definition. Only her first-person definition text reproduces her saved Raleigh numbers, so A uses that.
- **Step 5.** Human check of topics. **This is the labeling. Not done yet.**
- **Step 6.** Stars → signal (1★ = −1, 2–4★ = 0, 5★ = +1). Store × dimension mean, pooled with her hierarchical Bayesian model (PyMC).
- **Step 7.** Tract score = stores weighted by exp(−miles). Population-weighted regression on % Black and income.
- Two versions: one refit on NC, one "frozen" that runs Teresa's saved Raleigh model on NC reviews.
- Steps 1–4 don't depend on labels. Labels enter at step 6.

Pipeline B, one fix per problem, each tested against A (code: `NC Pipeline Improved/`, reads A's outputs, never writes them):

| Problem in A | Fix in B | Result (provisional) |
|---|---|---|
| One topic per review, but 79% of multi-sentence reviews cover 2+ subjects | Split into sentences; a review counts toward every dimension its sentences touch | Finds 70% of answer-key labels vs 45% |
| Definitions file selection under Acceptability, so Availability is rarely found | New definitions from the raters' rules | Availability recall 0.50 → 0.81 |
| Stars can't tell "good selection, high prices" | Sentiment read from each sentence's text | Right on 84% vs stars 64% |
| Stores with one 5★ review aren't shrunk and top the rankings | Better pooling (count model, county and store-type levels) | Halves prediction error for stores with ≤5 reviews |
| exp(−miles) is an urban rule; statewide z compares Boone with Charlotte | Real drive times on OSM roads; market = stores within 16 min | Shows who has no store in reach |

- B1 (removing the mood direction) and B3 (data-driven dimensions) were tested but are not in the final B chain.
- Caveat: B's accuracy numbers use Claude's stand-in answer key, coded with the same rules as B's definitions. They flatter B until people relabel.

## Results so far (all provisional)

- **Replication works.** Teresa's model on our newer reviews ranks her 42 matched stores the same way: rank correlation 0.90–0.95 across the five dimensions.
- **Her Table 5.3** comes back from her own store scores: Affordability +1.65 vs her +1.64. Her tract list is unknown, so it's approximate.
- **Statewide, distance and quality point opposite ways.** Tracts with a higher Black share are closer to a store, but the stores they reach score lower on every dimension. B makes the gaps larger.

| Coefficient on % Black (share) | A: ~1 mile | B: 16-min drive |
|---|---|---|
| Availability | −0.36 | −0.68 |
| Accessibility | −0.61 | −1.69 |
| Accommodation | −0.37 | −1.02 |
| Affordability | −0.24 | −1.31 |
| Acceptability | −0.71 | −1.13 |
| Distance / drive time to nearest | −1.03 | −1.33 |

- All A coefficients p < 0.001. R² is small (0.01–0.08).
- 489,418 people (162 tracts) have no study store within a 16-min drive. 34,787 have none within 32 min.
- **Raleigh, 8-min markets:** both pipelines agree higher-Black-share tracts reach stores that score lower on Availability, Accommodation, Acceptability. Teresa's positive Affordability result reverses or vanishes once every Wake store is included.
- **How many dimensions?** Store performance collapses to two (overall quality, price). What people talk about has more structure. The strongest theme the five A's don't name is checkout.

## What's wrong with the data (worst first)

1. Answer key isn't human yet. Every accuracy number measures agreement with Claude. Topic validity list is a hand-made draft.
2. Small samples on four dimensions. Median store: 60 Acceptability reviews, 5 Accommodation, 7 Accessibility.
3. Teresa's pooling rewards tiny samples (SE floored at 1e-6). A keeps it on purpose. B fixes it.
4. 257 big stores hold only their newest reviews (1,500 to ~5,600). They are a quarter of all reviews. Flagged.
5. Dates are vague ("3 years ago"). Good to about a year. No monthly analysis.
6. Reviewers have no ID, only a display name. No reviewer-level analysis.
7. Who posts reviews isn't who shops. Size of that bias is unmeasured.
8. Rural stores may still be missing. Census county check not done.
9. Store list includes judgment calls (1,001 dollar stores, 70 supplement stores) and plain errors: 3 out-of-state stores (IDs 5379, 9368, 8852) and 37 with a restaurant as main category. Still in the frozen data. Exclude them.
10. Only NC stores, so border tracts look worse than they are.
11. Short non-English reviews go through the English model untranslated (99.5% of 8+ word reviews are English).
12. Foot-traffic weights cover 2,806 of 4,452 stores and are licensed. Every result holds without them.

## What's left

Needs people (this week):
- [ ] Hand labeling, Wed Nov 18. Suyash runs it on his page (`docs/LABELING.docx`).
- [ ] Run `./run_after_labeling.sh` on the lab VM (~1.5 h). Produces final scores and rebuilds the Drive folder.
- [ ] Update `PIPELINE_GUIDE.md` and these docs with the human-labeled numbers.
- [ ] Upload the rebuilt folder to Drive, then hand the project to Caroline.
- [ ] Transfer the GitHub repo to Dr. Rachunok or a lab account.

Needs outside answers:
- [ ] Ask Teresa for her Raleigh tract list, the store set behind her distance-to-nearest, and whether her z-scores are before or after pooling (her data and thesis disagree; A follows the thesis).

Any time:
- [ ] Check store list against Census county counts (`Frame/reports/census_validation.csv` in the scraping repo). Half a day.
- [ ] Add out-of-state stores near the border to B6.
- [ ] Fix the 40 store-list errors. Requires a new freeze.
- [ ] Decide if the 32-min market is a sensitivity check or the rural fallback. It flips the Affordability sign.

Decisions for Dr. Rachunok:
- Headline pipeline. Recommendation: B as the main result, A beside it as the replication.
- Market size. Recommendation: 8 min for Raleigh, 16 min statewide.
- Is checkout a sixth dimension? The raters' "other subject" write-ins can answer it.

## Where things live

- **Lab VM** `manabe.ie.ncsu.edu`, `/home/scsapre/aspatial_food_access_research/` (~32 GB). Everything, including embeddings (~5 GB) and licensed data. The pipelines run here.
- **GitHub** `suyashh77/nc-aspatial-food-access` (private). Code only (~60 files). Should be transferred to Dr. Rachunok or a lab account.
- **Google Drive** `NC_Aspatial_Food_Access` (https://drive.google.com/drive/folders/1VH4wPZjRUApfQtznYghnRkXsL14XJbm9). Docs, slides, datasets, in the layout described in `README.docx`.
- **Suyash's PC.** Scraping code ("Gmaps Scraper" repo) and the official frozen file `Step_5_master_reviews_FROZEN_2026-10-24.db` with its manifest. Upload to Drive **without** `Auth/` (live Google logins).
- `data/database/Step_5_master_reviews.db` has the same content as the freeze but a different checksum (different SQLite version).

## For whoever runs the code

Setup from scratch:
- Clone the repo, then `./place_data.sh <path to this folder's data/>`. It checks the database checksums and puts every file where the code expects it (~2.5 GB).
- Don't run `place_data.sh` on the VM. Everything is already in place there.
- Python env: `requirements_lab_vm.txt`. Steps 1–3 need RAPIDS cuML and a CUDA GPU. VM has an RTX A4500 (20 GB).
- Tests: `cd "NC Pipeline" && .venv/bin/python -m pytest -q tests` (14 tests).

Rules:
- Never write to the review databases. Code opens them read-only. A data change means a new freeze and checksum.
- Never delete reviews or stores. Set aside with a reason.
- Pipeline A stays Teresa's method. Improvements go in B, one switch each, tested against A.
- Pipeline B never writes into `NC Pipeline/`.
- Licensed Advan/SafeGraph store-level rows never leave the VM. Tract aggregates are fine.
- Update `PIPELINE_GUIDE.md` with a changelog line after any pipeline change.

Known traps:
- `step6` keeps Teresa's SE floor (1e-6) on purpose. Don't fix it in A.
- Teresa's `store_scores.csv` `_z` is z of raw means; her thesis says pooled. A follows the thesis.
- `b6_local_market.py --weights visits` and default `b_raleigh.py` need licensed data, so they only run on the VM. Use `--weights equal` elsewhere.
- B6 drive times are free-flow (no turns or signals), slightly optimistic.
- Ignore `*haywood*` files. Haywood County was dropped 2026-11-11.
- `config.py` docstring says steps 4–5 aren't built. They are.
- Don't scrape again without reading `docs/scraping/STATUS.md` §6.

Pipeline commands, A (run from `NC Pipeline/`):

```
step1_embed.py         clean + embed           → data/step1_embeddings.npy
step2_umap.py          UMAP 5-D                → data/step2_umap5.npy
step3_topics.py fit    HDBSCAN + BERTopic      → data/step3_*  (83 topics)
step4_match.py         topic → dimension       → data/step4_review_dims.parquet
step4_frozen.py        Teresa's Raleigh model  → data/step4_review_dims_frozen.parquet
step6_store_scores.py  [--arm frozen] [--stores raleigh --tag raleigh]
step7_tracts.py        [--arm frozen]
export_review_table.py / export_poi_table.py
```

Pipeline B (run from `NC Pipeline Improved/` with A's venv):

```
b2_sentences.py        reviews → 2.0M sentences → 83 sentence topics
b4_definitions.py      definitions from raters' codebook, tuned on gold
b7_text_sentiment.py   sentence sentiment classifier
b_pipeline.py          all switches on → out/b_store_scores.csv
b6_roads.py, b6_drive_times.py   OSM drive times to 32 min
b6_local_market.py --scores a|b5|b [--weights visits|equal]
b_raleigh.py [--weights equal]
```

Rebuild the Drive folder: `"NC Pipeline/.venv/bin/python" build_drive_package.py [--zips]`. Commit first; it snapshots the repo.
