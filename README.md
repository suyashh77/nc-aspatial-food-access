# NC aspatial food access

Every grocery store in North Carolina (4,442), the Google reviews written about them (1.94 million), and scores for each store and census tract on the five dimensions of food access. The method scales Teresa Groton's Raleigh thesis to the whole state. NC State ISE, Dr. Benjamin Rachunok's group. Handed off by Suyash Sapre on November 15, 2026.

This repository is the starting point: code, handoff documents, results and data in one place.

## Get started

1. Clone this repo.
2. Get the large files (too big for git) from the Google Drive folder and copy them in:

| From Drive | Put it in the repo at |
|---|---|
| `9_Database/` — all five `.db` files | `Reviews Data/` |
| `9_Database/step1_reviews.parquet` | `NC Pipeline/data/` |
| `4_Review_Level/NC_reviews_with_dimensions_PROVISIONAL.parquet` | `NC Pipeline/data/exports/NC_reviews_with_dimensions.parquet` (drop `_PROVISIONAL` from the name) |

   Shortcut if you use the GitHub CLI: `./fetch_data.sh` downloads the same files from this repo's release and puts them in place.

Then read the guide written for you:

| You are | Read |
|---|---|
| Dr. Rachunok, or anyone who wants the whole story and the data's weak spots | `handoff/FOR_DR_RACHUNOK.md` |
| Running the hand labeling, or using the scores in a study | `handoff/FOR_CAROLINE.md` |
| A Claude session or a programmer picking up the code | `handoff/FOR_CLAUDE.md` (Claude reads `CLAUDE.md` automatically) |

The presentation covering the whole project: `handoff/presentation/Final_Presentation.html` (open in a browser; arrow keys, N for speaker notes) or `Final_Presentation.pdf`.

## Status

All scores are **provisional** until the hand labeling is done (planned for the week of November 16, 2026). After it, `./run_after_labeling.sh --csv <folder>` produces the final numbers. See `handoff/FOR_CAROLINE.md`.

## What's here

| Path | Contents |
|---|---|
| `handoff/` | The three guides, data dictionary, file manifest, presentation |
| `NC Pipeline/` | Pipeline A, Teresa's method at NC scale: code, tests, decision tests, labeling page, and its outputs in `data/` |
| `NC Pipeline Improved/` | Pipeline B, tested improvements: code and outputs in `out/` |
| `PIPELINE_GUIDE.md` | The full method, every decision test and every number |
| `run_after_labeling.sh` | Reruns everything once the hand labels are in |
| `build_drive_package.py` | Builds a shareable folder of the final files (`google_drive_upload/`) |
| `Reviews Data/` | The scraping project's documentation (the databases arrive with `fetch_data.sh`) |
| `Teresa Project/` | The parts of Teresa's pilot the code reads, her notebooks and her thesis |

Not included: the Python environment (exact versions in `requirements_lab_vm.txt`; it also needs RAPIDS cuML and a CUDA GPU for steps 1–3), about 4.6 GB of embeddings that the pipeline scripts rebuild, and licensed foot-traffic data (Advan, SafeGraph), which stays on the lab server. The scraping code itself lives in a separate repository on Suyash's machine; its documentation is in `Reviews Data/`.

Private research data. Don't share outside the research group.
