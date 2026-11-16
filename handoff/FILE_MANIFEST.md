# What lives where: GitHub, Google Drive, and the server

Everything for this project is on the lab VM at `manabe.ie.ncsu.edu:/home/scsapre/aspatial_food_access_research/` (about 32 GB). From there it is split in two. GitHub holds the code. Google Drive holds the documents and reports, the presentation, and the datasets.

Large files that a script can regenerate stay on the server. Licensed data never leaves it. This page lists every top-level folder and where it goes.

## GitHub: the code

The private repo `suyashh77/nc-aspatial-food-access` holds the scripts for both pipelines, the tests, the labeling page, the scripts that build the Drive folder and the slides, `PIPELINE_GUIDE.md`, and the docs in `handoff/` (Markdown sources of the Word files in Drive). That is about 70 files and 1 MB. Its `.gitignore` is an allowlist: nothing new is committed unless it's listed there, so result files can't slip in.

Until November 16, 2026 the repo also held the result files and documents. They were moved to Drive. Old commits still contain them.

## Google Drive: upload the whole `google_drive_upload/` folder

`build_drive_package.py` builds it. With `--zips` it also writes `drive_zips/docs.zip` and `drive_zips/data.zip`, which are easier to download from the VM. (Before November 16, 2026 the Drive folder used `1_Docs_and_Reports`, `2_Presentation` and `3_Datasets`.)

| In the Drive folder | Comes from | Size |
|---|---|---|
| `README.docx`, `LABEL_STATUS.txt` | `handoff/README.md`, the build | small |
| `docs/` (handover, labeling, using the data, data dictionary, pipeline guide, scraping record, all as Word; Teresa's thesis; `markdown/` sources plus `FOR_CLAUDE.md` and this manifest) | `handoff/`, `PIPELINE_GUIDE.md`, `Reviews Data/*.md` | 8 MB |
| `presentation/` (the final slides, web page and PDF) | `handoff/presentation/` | 8 MB |
| `labeling/` (the labeling page, its inputs, imported labels) | `NC Pipeline/apps/web_labeler/`, `decisions/` | 1 MB |
| `results/` (decision tests, Pipeline B tests, Raleigh case study workbook) | test results, `NC Pipeline Improved/out/b_raleigh*` | 1 MB |
| `data/store_list/` | `NC Pipeline/data/exports/` | 3 MB |
| `data/scores/` (store and tract scores, both pipelines, regressions) | `NC Pipeline/data/`, `NC Pipeline Improved/out/` | 15 MB |
| `data/review_level/` (every review with labels) | `NC Pipeline/data/exports/` | 120 MB |
| `data/raleigh_case_study/` | `NC Pipeline Improved/out/b_raleigh*` | 1 MB |
| `data/database/` (the review database, side databases, and the cleaned reviews the pipelines read) | `Reviews Data/`, `NC Pipeline/data/` | 0.9 GB |
| `data/pipeline_working_files.zip` (the result and working files the code reads, at their repo paths: topic models, UMAP coordinates, all outputs, Census tables, Teresa's files) | both pipelines, `Teresa Project/`, `handoff/` | about 0.2 GB |
| `data/PACKAGE_CONTENTS.csv` (every file with its checksum) | the build | small |
| `code/` (a backup zip of the GitHub repo) | the repo | 1 MB |

A fresh clone of the repo plus `./place_data.sh <data>` gives the same working folder as the VM, minus the items under "Stays on the server".

## Upload separately from your PC

| What | Where | Why |
|---|---|---|
| The scraping code ("Gmaps Scraper" repo: `Frame/`, `Scraper_v2/`, `Remediation/`, `Deploy/`, docs) | Suyash's PC | It built the store list and the reviews, and it isn't on the VM. **Zip it without `Auth/`** (live Google logins) and without `Data/` (the databases are already in `3_Datasets/database/`) |
| The frozen file `Step_5_master_reviews_FROZEN_2026-10-24.db` and `Remediation/master_freeze_manifest_v2.json` | Suyash's PC | The official freeze with its published checksum. `3_Datasets/database/` holds a copy with the same content but a different checksum |

## Stays on the server

| Folder | Size | Why it stays |
|---|---|---|
| `NC Pipeline/.venv/` | 3.5 GB | The Python environment. Rebuild from `requirements_lab_vm.txt` in the repo |
| `NC Pipeline/data/step1_embeddings.npy` | 1.4 GB | Review embeddings. `step1_embed.py` rebuilds them in about 10 minutes on the GPU |
| `NC Pipeline Improved/out/b2_sentence_embeddings.npy` | 3.0 GB | Sentence embeddings. `b2_sentences.py` rebuilds them |
| Other intermediate files in `NC Pipeline/data/` and `NC Pipeline Improved/out/` (UMAP coordinates, topic model, grid search, drive-time graph) | about 1 GB | Inputs to later stages, all regenerable. They are in `pipeline_working_files.zip` too, because rerunning after labeling needs them |
| `NC Pipeline Improved/data_external/` (OpenStreetMap road extract) | 411 MB | Public download (Geofabrik NC extract) |
| `NC Pipeline/data_external/` (Census tables) | small | Public download (ACS 2022, 2022 gazetteer) |
| `grocery-foot-traffic/` | 390 MB | **Licensed** Advan and SafeGraph data. Its README says keep store-level rows inside the project |
| `Reviews Data/Step_6_grocery_pois.db` | 271 MB | Dewey (SafeGraph) places list, licensed. Its useful columns are already in the store list |
| `Reviews Data/Step_1.*`, `Step_2.0`, `Step_3.*` | 0.4 GB | The spring 2026 first attempt (Overture download, Outscraper results, old classifications). Replaced by the September rebuild; kept for history |
| `Reviews Data/Step_5_master_reviews_FROZEN_2026-10-11.db` | 582 MB | Freeze v1, superseded by v2 |
| `Teresa Project/` | 752 MB | Teresa's original notebooks, model and data. The parts the pipelines read (her notebooks, saved Raleigh model, store scores) and her thesis are in Drive. The rest should already be with the group from Teresa. Upload it only if the group doesn't have a copy |
| `geomatch/` | 22 GB | The matching library. The code is on GitHub (github.com/suyashh77/geomatch). `data_external/` (14 GB) is downloaded source data, and `experiments/` holds model runs |
| `logs/`, `NC Pipeline/logs/` | small | Run logs |
| `google_drive_upload.prev/` | 1 GB | The previous build, kept by the build script in case a rebuild goes wrong |
| `drive_zips/` | 1 GB | The zips made for uploading by hand. Rebuilt by `build_drive_package.py --zips` |
| `_superseded/` | 11 MB | The November 13 handoff draft and its upload folder. Replaced by this package |

## Cleanup done at handoff (November 15)

- Moved the November 13 draft (`GOOGLE_DRIVE_HANDOFF_README.md` and the old `google_drive_upload/`) to `_superseded/`. It quoted some numbers wrongly, called human labels finished when none existed yet, and included personal details that don't belong in a shared folder.
- Replaced the November 13 Raleigh talk guide. It overstated the replication ("perfect", "the exact same rankings") and called Teresa's Affordability result an artifact, which goes further than the evidence. The Raleigh section of *FOR_CAROLINE* replaces it, and the Raleigh workbook is now rebuilt from the results files on every build.
- Deleted Python caches (`__pycache__/`, `.pytest_cache/`).

- November 16: split the handoff into GitHub (code) and Drive (documents, slides, datasets). The result files stopped being tracked in git but stayed on disk. Removed `fetch_data.sh` (replaced by `place_data.sh`) and the single zip of the earlier Drive layout.

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
