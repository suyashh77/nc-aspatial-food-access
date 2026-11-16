# What goes to Google Drive, and what stays

Everything for this project lives on the lab VM at `manabe.ie.ncsu.edu:/home/scsapre/aspatial_food_access_research/` (about 32 GB). The Drive copy is `google_drive_upload/` (about 1 GB), built by `build_drive_package.py`. This page lists every top-level folder and why it does or doesn't go to Drive.

The rule: Drive gets everything a person needs to use, check or rerun the work. Large files that a script can regenerate stay on the server. Licensed data never leaves it.

## The GitHub repo (the starting point)

The private repo `suyashh77/nc-aspatial-food-access` holds the code, the handoff documents, every result file and Teresa's files the code reads. Files over GitHub's 100 MB limit (the five databases, the cleaned reviews and the review export) are attached to its `handoff-2026-11-15` release, and `./fetch_data.sh` puts them in place. Its `.gitignore` is an allowlist: nothing new is committed unless it's listed there. The repo leaves out the same things Drive does (licensed data, the Python environment, regenerable embeddings).

## Goes to Drive (upload the whole `google_drive_upload/` folder)

| In the Drive folder | Comes from | Size |
|---|---|---|
| `README` | `handoff/README.md` | small |
| `1_Read_Me_First/` (the three guides, data dictionary, this manifest, the final presentation) | `handoff/` | 5 MB |
| `2_Store_List/` | `NC Pipeline/data/exports/` | 3 MB |
| `3_Scores/` (store and tract scores, both pipelines, regressions) | `NC Pipeline/data/`, `NC Pipeline Improved/out/` | 15 MB |
| `4_Review_Level/` (every review with labels) | `NC Pipeline/data/exports/` | 120 MB |
| `5_Human_Labeling/` (the labeling page, its inputs, imported labels) | `NC Pipeline/apps/web_labeler/`, `decisions/` | 1 MB |
| `6_Raleigh_Case_Study/` | `NC Pipeline Improved/out/b_raleigh*` | 1 MB |
| `7_Methods_and_Reference/` (pipeline guide, scraping record, decision tests, Teresa's thesis) | various | 10 MB |
| `8_Code/nc-aspatial-food-access_repo.zip` (a full copy of the GitHub repo: code, docs, results, Teresa's files, pinned package versions) | the repo | about 0.3 GB |
| `9_Database/` (the review database, side databases, and the cleaned reviews the pipelines read) | `Reviews Data/`, `NC Pipeline/data/` | 1 GB |

## Upload separately from your PC

| What | Where | Why |
|---|---|---|
| The scraping code ("Gmaps Scraper" repo: `Frame/`, `Scraper_v2/`, `Remediation/`, `Deploy/`, docs) | Suyash's PC | It built the store list and the reviews, and it isn't on the VM. **Zip it without `Auth/`** (live Google logins) and without `Data/` (the databases are already in `9_Database/`) |
| The frozen file `Step_5_master_reviews_FROZEN_2026-10-24.db` and `Remediation/master_freeze_manifest_v2.json` | Suyash's PC | The official freeze with its published checksum. `9_Database/` holds a copy with the same content but a different checksum |

## Stays on the server

| Folder | Size | Why it stays |
|---|---|---|
| `NC Pipeline/.venv/` | 3.5 GB | The Python environment. Rebuild from `requirements_lab_vm.txt` in the code zip |
| `NC Pipeline/data/step1_embeddings.npy` | 1.4 GB | Review embeddings. `step1_embed.py` rebuilds them in about 10 minutes on the GPU |
| `NC Pipeline Improved/out/b2_sentence_embeddings.npy` | 3.0 GB | Sentence embeddings. `b2_sentences.py` rebuilds them |
| Other intermediate files in `NC Pipeline/data/` and `NC Pipeline Improved/out/` (UMAP coordinates, topic model, grid search, drive-time graph) | about 1 GB | Inputs to later stages, all regenerable. Needed on the server to rerun after labeling |
| `NC Pipeline Improved/data_external/` (OpenStreetMap road extract) | 411 MB | Public download (Geofabrik NC extract) |
| `NC Pipeline/data_external/` (Census tables) | small | Public download (ACS 2022, 2022 gazetteer) |
| `grocery-foot-traffic/` | 390 MB | **Licensed** Advan and SafeGraph data. Its README says keep store-level rows inside the project |
| `Reviews Data/Step_6_grocery_pois.db` | 271 MB | Dewey (SafeGraph) places list, licensed. Its useful columns are already in the store list |
| `Reviews Data/Step_1.*`, `Step_2.0`, `Step_3.*` | 0.4 GB | The spring 2026 first attempt (Overture download, Outscraper results, old classifications). Replaced by the September rebuild; kept for history |
| `Reviews Data/Step_5_master_reviews_FROZEN_2026-10-11.db` | 582 MB | Freeze v1, superseded by v2 |
| `Teresa Project/` | 752 MB | Teresa's original notebooks, model and data. The pipelines read it. Her thesis PDF goes to Drive; the rest should already be with the group from Teresa. Upload it only if the group doesn't have a copy |
| `geomatch/` | 22 GB | The matching library. The code is on GitHub (github.com/suyashh77/geomatch). `data_external/` (14 GB) is downloaded source data, and `experiments/` holds model runs |
| `logs/`, `NC Pipeline/logs/` | small | Run logs |
| `google_drive_upload.prev/` | 1 GB | The previous build, kept by the build script in case a rebuild goes wrong |
| `_superseded/` | 11 MB | The November 13 handoff draft and its upload folder. Replaced by this package |

## Cleanup done at handoff (November 15)

- Moved the November 13 draft (`GOOGLE_DRIVE_HANDOFF_README.md` and the old `google_drive_upload/`) to `_superseded/`. It quoted some numbers wrongly, called human labels finished when none existed yet, and included personal details that don't belong in a shared folder.
- Replaced the November 13 Raleigh talk guide. It overstated the replication ("perfect", "the exact same rankings") and called Teresa's Affordability result an artifact, which goes further than the evidence. The Raleigh section of *FOR_CAROLINE* replaces it, and the Raleigh workbook is now rebuilt from the results files on every build.
- Deleted Python caches (`__pycache__/`, `.pytest_cache/`).

Nothing else was deleted.

## Could be deleted later (not done)

These are safe to delete because a script rebuilds them, but deleting them saves only disk space, and rerunning after labeling needs most of them. Wait until the final scores are in.

| What | Size | Rebuilt by |
|---|---|---|
| `geomatch/data_external/` | 14 GB | geomatch's download scripts |
| `NC Pipeline Improved/out/b2_sentence_embeddings.npy` | 3.0 GB | `b2_sentences.py` |
| `NC Pipeline/data/step1_embeddings.npy` | 1.4 GB | `step1_embed.py` |
| `NC Pipeline/data/step3_grid/`, `exp_no_empty/` | 117 MB | `step3_topics.py grid`, `exp_no_empty.py` |
| Haywood County outputs (`NC Pipeline/data/*haywood*`, `data/exports/haywood/`) | 1 MB | Haywood was dropped from scope on November 11 |
