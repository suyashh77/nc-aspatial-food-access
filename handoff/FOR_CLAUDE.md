# Technical handoff: NC aspatial food access

For a Claude session (or a person) picking up the code. Written 2026-11-15 at handoff, when the project's owner, Suyash Sapre, left the project. Read this whole file before running anything.

People-facing docs: `FOR_DR_RACHUNOK.md` (project, methods, data limits) and `FOR_CAROLINE.md` (hand labeling, using the data). The full method, every decision test and every number: `../PIPELINE_GUIDE.md`.

**Home of the project:** the private GitHub repo `suyashh77/nc-aspatial-food-access` (owner may change if it's transferred to the lab). Clone it, run `./fetch_data.sh` for the databases and large review files, and the layout below matches. Package versions: `requirements_lab_vm.txt`.

## State at handoff

- Data collection is closed. The review database is frozen (freeze v2, 2026-10-24).
- Pipeline A (Teresa Groton's method, NC scale) and Pipeline B (tested improvements) run end to end.
- **Every score is provisional.** It uses a draft list of valid topics (`NC Pipeline/data/parents_draft.csv`) and a stand-in answer key labeled by Claude (`NC Pipeline/decisions/out/gold_labels_part*.txt`).
- Hand labeling happens on a web page (Section 4). When it's done, `run_after_labeling.sh` produces the final outputs. That script was tested end to end on 2026-11-15 with five fake raters, in a copy of the project.

## Layout

All paths are relative to `/home/scsapre/aspatial_food_access_research/` on the lab VM (`manabe.ie.ncsu.edu`).

| Path | What |
|---|---|
| `Reviews Data/Step_5_master_reviews.db` | The dataset (SQLite, ~700 MB). Content equals freeze v2. **Open read-only.** |
| `Reviews Data/Step_5_{out_of_scope,closed_stores,excluded_stores}.db` | Stores ruled out, with their reviews and reasons |
| `Reviews Data/*.md` | Copies of the scraping project's docs (`STATUS.md` is the full record) |
| `NC Pipeline/` | Pipeline A: `step1`–`step7` scripts, `config.py`, `decisions/` (tests D1–D10), `apps/` (labelers), `tests/` |
| `NC Pipeline Improved/` | Pipeline B: `b1`–`b7` switches, `b_pipeline.py` (all switches on), `b_raleigh.py`. Reads A's outputs, never writes them |
| `Teresa Project/` | Teresa's notebooks, her saved Raleigh model, her data and thesis. Read-only reference |
| `grocery-foot-traffic/` | Advan foot traffic and SafeGraph spend. **Licensed: never copy store-level rows off the server** |
| `geomatch/` | The record-matching library used to build the store list (code also at github.com/suyashh77/geomatch). Its `data_external/` is 14 GB of downloaded source data |
| `run_after_labeling.sh` | Everything to rerun after labeling, in order |
| `build_drive_package.py` | Builds `google_drive_upload/` from current outputs |
| `handoff/` | These docs, the data dictionary, the file manifest, the final presentation |
| `PIPELINE_GUIDE.md` | The living method document. Update it, with a changelog line, after any pipeline change |

The scraping code (store frame, scraper, repair scripts r00–r18) is **not on this VM**. It lives on the owner's PC as the "Gmaps Scraper" repo, and its docs are copied into `Reviews Data/`. Don't scrape again without reading `Reviews Data/STATUS.md` §6.

## Rules

1. **Never write to the review databases.** Code opens them with `?mode=ro`. A change to the data means a new freeze, a new checksum and an updated `Reviews Data/STATUS.md`.
2. **Never delete reviews or stores.** Set aside with a reason (the scraping project's rule).
3. **Pipeline A stays Teresa's method.** Change it only where NC's scale forces a change, and document why in `PIPELINE_GUIDE.md`. Improvements go in Pipeline B as one switch each, tested against A.
4. **Pipeline B never writes into `NC Pipeline/`.**
5. **Licensed data stays on the server.** Advan and SafeGraph store-level rows don't go to Drive, email or any outside service. Aggregates (tract averages weighted by visits) are fine.
6. **Tests stay green:** `cd "NC Pipeline" && .venv/bin/python -m pytest -q tests` (14 tests).
7. Use the project's Python: `"NC Pipeline/.venv/bin/python"`. It has cuML (GPU UMAP/HDBSCAN), PyMC with nutpie, BERTopic and sentence-transformers. The VM has an RTX A4500 (20 GB) and usually under 35 GB of free RAM.

## The pipelines in one screen

Pipeline A (`NC Pipeline/`, run from that folder):

```
step1_embed.py            clean text (Teresa's function) + all-MiniLM-L12-v2 → data/step1_embeddings.npy (975,388 × 384)
step2_umap.py             UMAP 5-D (cuML)                                 → data/step2_umap5.npy
step3_topics.py fit       HDBSCAN min cluster 2,000 + BERTopic → 83 topics → data/step3_*
step4_match.py            topic centroid vs first-person definitions, softmax τ 0.11, extra labels within δ 0.17
                                                                          → data/step4_review_dims.parquet
step4_frozen.py           Teresa's Raleigh model on NC reviews            → data/step4_review_dims_frozen.parquet
step5_*                   hand check: validation table, label import
step6_store_scores.py     star signal, store × dimension mean, PyMC partial pooling (her model and SE floor)
                          [--arm frozen] [--stores raleigh --tag raleigh]  → data/step6_store_scores*.csv
step7_tracts.py           exp(−miles) tract scores, population-weighted WLS on % Black + income [--arm frozen]
                                                                          → data/step7_tracts_*.csv, step7_regressions_*.csv
export_review_table.py    one row per review with every pipeline's labels → data/exports/NC_reviews_with_dimensions.parquet
export_poi_table.py       the store table from the frozen DB              → data/exports/NC_Grocery_POIs_v2.xlsx
```

Steps 1–4 don't depend on the hand labels and never need rerunning unless the method changes. Validity enters at stage 6 (`step6_store_scores.validity()`) and in `decisions/common.invalid_topics()`.

Pipeline B (`NC Pipeline Improved/`, run from that folder with A's venv):

```
b2_sentences.py         reviews → 2.0M sentences → 83 sentence topics (out/b2_*)
b4_definitions.py       definitions from the raters' codebook, tuned on the gold set → out/b4_definition_vectors.npz
b7_text_sentiment.py    sentence sentiment classifier on the embeddings  → out/b7_sentence_sentiment.npy
b_review_dims.py        review × dimension labels (shared by b_pipeline and the export)
b_pipeline.py           all switches on, B5 pooling (county + category levels) → out/b_store_scores.csv, out/b_review_dims.parquet
b6_roads.py, b6_drive_times.py   OSM road graph, tract → store drive times to 32 min → out/b6_drive_times.parquet
b6_local_market.py --scores a|b5|b [--weights visits|equal]   16/32-minute markets → out/b6_tracts_*.csv
b_raleigh.py [--weights equal]   Raleigh case study vs Teresa      → out/b_raleigh*.json, b_raleigh_stores*.csv
```

B1 (removing the mood direction) and B3 (data-driven dimensions) are tested switches whose results are in the guide. They are not in the final B chain.

## Hand labeling

**Labeling runs on Caroline's copy of the page**, published from her claude.ai account so she owns the results. Her link: _(Caroline adds it here)_. Suyash's original, https://claude.ai/artifact/7cCD2HedpJ97FgEmnLo2uH, was never used for labeling; ignore it. Source: `NC Pipeline/apps/web_labeler/` (`labeler.html` + `data.js` + `data_b2.js`; `build_data*.py` regenerate the data files). It uses the artifact database, user identity and downloads.

Database layout (one document per rater, keyed by the rater's account id):

| Collection | Document body |
|---|---|
| `gold/<uid>` | `{code, labels: {<gid>: {id, Availability: "+"/"-"/"m", ..., none, other, conf, notes, sec, at}}}` |
| `topics/<uid>` | `{code, topics: {<topic>: {group, status, reason}}, groups: {<key>: {name, members, status, reason, scores, notes, at}}}` |
| `topics_b2/<uid>` | same as `topics`, for Pipeline B's sentence topics |
| `claims/<uid>` | `{gids}`: which reviews each rater took, so the page can hand out reviews 5 raters at a time |

At handoff all collections were empty. Only the page's owner sees the Results tab, and only someone with access to the page can pull its database, so run the pull from Caroline's account or use her Results-tab CSVs.

## After labeling

Pull the labels one of two ways.

**From the database** (a Claude session signed in as the page's owner):

```
ArtifactData list  url=<Caroline's page>  collection=gold       out_dir="/home/scsapre/aspatial_food_access_research/NC Pipeline/data/web_labels"
ArtifactData list  url=<Caroline's page>  collection=topics     out_dir=(same)
ArtifactData list  url=<Caroline's page>  collection=topics_b2  out_dir=(same)
```

Page with `query.cursor` if a list returns `next_cursor`. Files land as `data/web_labels/<collection>/<uid>.json`, which is what the import expects. Then:

```bash
./run_after_labeling.sh --json "NC Pipeline/data/web_labels"
```

**From the Results-tab CSVs** (no Claude needed): put `gold_labels_all_raters.csv`, `step5_manual_scores_all_raters.csv` and optionally `b2_manual_scores_all_raters.csv` in one folder, then `./run_after_labeling.sh --csv <folder>`. Both routes give identical outputs (checked 2026-11-15; `tests/test_import_labels.py`).

What the import writes:

| File | Read by |
|---|---|
| `NC Pipeline/decisions/labels/<initials>.csv`, `gold_all_raters.csv` | people |
| `NC Pipeline/decisions/labels/gold_consensus.csv` | `decisions/common.gold()`: used instead of Claude's stand-in once it covers ≥200 reviews with ≥3 raters |
| `NC Pipeline/data/step5_manual_scores.csv` (+ one per rater) | stage 6 and `common.invalid_topics()`: only `manual_topic_status == "valid"` topics count |
| `NC Pipeline Improved/out/b2_manual_scores.csv` | `b_review_dims.py`: replaces the automatic generic filter |
| `NC Pipeline/data/step5_import_meta.json` | raters, counts, Fleiss' kappa |

The script then reruns stages 6–7 (both arms, state and Raleigh), D1–D10, B4, `b_pipeline.py`, B6 (four variants), `b_raleigh.py` (two), the review export and the Drive package. About 1.5 hours; `--with-b5` adds the B5 holdout test (about an hour more). Logs go to `logs/after_labeling_<time>/`.

Then:

1. Check `google_drive_upload/LABEL_STATUS.txt` says HUMAN.
2. Read `step5_import_meta.json`. A Fleiss' kappa under 0.4 on a dimension means the raters disagree, and the result for that dimension needs a caveat.
3. Update `PIPELINE_GUIDE.md`. Every table there was computed on provisional labels. Replace the numbers from the json files in `NC Pipeline/decisions/out/` and `NC Pipeline Improved/out/`, mark them as human-labeled, and add a changelog line. B4's caveat (stand-in gold coded with the same rules as B4's text) goes away once the gold is human.
4. Update the people-facing docs in `handoff/` where they quote numbers, then rerun `build_drive_package.py`.

## Rebuilding the Drive folder

```bash
"NC Pipeline/.venv/bin/python" build_drive_package.py
```

It builds a fresh `google_drive_upload/`, keeps the previous one as `google_drive_upload.prev/`, converts the `handoff/*.md` docs to Word with pandoc, rebuilds the Raleigh workbook from `b_raleigh*.json`, zips the code with pinned package versions, copies the databases (reusing an unchanged copy by checksum), and writes `PACKAGE_CONTENTS.csv` with checksums. Score files are named `*_PROVISIONAL` until stage 6 reports human validity. `handoff/FILE_MANIFEST.md` says what goes to Drive and why.

## Known traps

- `Reviews Data/STATUS.md` names the frozen file `Step_5_master_reviews_FROZEN_2026-10-24.db`. That file is on the owner's PC, not here. The VM's `Step_5_master_reviews.db` has the same content (same counts; the checksum recorded in `NC Pipeline/data/step1_meta.json` at embedding time still matches). A `VACUUM INTO` copy does not reproduce the freeze's published checksum byte for byte, because the SQLite versions differ.
- `step6` keeps Teresa's SE floor (1e-6) on purpose. Don't "fix" it in A; B5 is the fix.
- Teresa's `store_scores.csv` `_z` column is the z of raw means, while her thesis says pooled. A follows the thesis (D8).
- `parents_draft.csv` is a hand-made draft. It decides provisional validity, nothing else.
- B6 drive times are free-flow (no turn or signal delays) and only count NC stores, so border tracts look worse than they are.
- `b6_local_market.py --weights visits` reads `grocery-foot-traffic/stores.csv` (licensed). `--weights equal` doesn't, and Teresa's method uses no foot traffic.
- The `step6_*_haywood*` and `data/exports/haywood/` files are from Haywood County, which was dropped from scope on 2026-11-11. Ignore them.
- `NC Pipeline/config.py`'s docstring still says steps 4–5 are "not built yet". They are; this file and the guide are current.
- The old `GOOGLE_DRIVE_HANDOFF_README.md` (Nov 13) was superseded and moved to `_superseded/`. Some of its numbers were wrong and it held personal details. Don't use it.

## Open work, roughly in order

1. Hand labeling, then `run_after_labeling.sh`, then refresh the guide and docs.
2. Ask Teresa for her Raleigh tract list, her distance-to-nearest store set, and raw vs pooled z.
3. Check the store list against Census county counts (`Frame/reports/census_validation.csv`, in the scraping repo).
4. Add out-of-state stores near the border to B6.
5. Store-list errors found at handoff: 3 stores with out-of-state addresses (5379, 9368, 8852) and 37 with a restaurant as the main Google category. Fixing them means a new freeze (see Rules); until then, analyses should exclude them.
6. Name the sixth-dimension candidates (checkout is strongest) from the raters' "other subject" write-ins and a B3 rerun on human labels.
