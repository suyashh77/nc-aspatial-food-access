# NC aspatial food access: code

Every grocery store in North Carolina (4,442), the Google reviews written about them (1.94 million), and scores for each store and census tract on the five dimensions of food access. The method scales Teresa Groton's Raleigh thesis to the whole state. NC State ISE, Dr. Benjamin Rachunok's group. Handed off by Suyash Sapre on November 15, 2026.

The project lives in two places:

| Where | What |
|---|---|
| **This repository** | The code: both pipelines, the tests, the labeling page, the scripts that build the Drive folder and the slides |
| **The Google Drive folder** `NC_Aspatial_Food_Access` _(add the link here)_ | `1_Docs_and_Reports` (the guides, the method, the data's weak spots), `2_Presentation` (the slides), `3_Datasets` (store list, scores, reviews, databases) |

If you only want to read about the project or use the data, you don't need this repository. Open the Drive folder and start with its `README`.

## Get started with the code

1. Clone this repo.
2. Download `3_Datasets` from the Drive folder.
3. Run `./place_data.sh <path to 3_Datasets>`. It checks the databases, then puts every result file, working file and document where the code expects it (about 2.5 GB).
4. Read `handoff/FOR_CLAUDE.md`. Claude reads `CLAUDE.md` on its own, which points there.

After step 3 the `handoff/` folder also holds the guides for people (`FOR_DR_RACHUNOK.md`, `FOR_CAROLINE.md`), the data dictionary and the file manifest.

## Status

All scores are **provisional** until the hand labeling is done (planned for the week of November 16, 2026). After it, `./run_after_labeling.sh --csv <folder>` produces the final numbers and rebuilds the Drive folder. The steps are in `FOR_CAROLINE` (Drive, `1_Docs_and_Reports`).

## What's here

| Path | Contents |
|---|---|
| `NC Pipeline/` | Pipeline A, Teresa's method at NC scale: scripts, tests, decision tests, the labeling page |
| `NC Pipeline Improved/` | Pipeline B, the tested improvements |
| `PIPELINE_GUIDE.md` | The full method, every decision test and every number |
| `handoff/FOR_CLAUDE.md` | The technical handoff: layout, rules, what to run after labeling |
| `run_after_labeling.sh` | Reruns everything once the hand labels are in |
| `build_drive_package.py` | Builds the Drive folder (`google_drive_upload/`) from the current outputs |
| `place_data.sh` | Puts the Drive data into a fresh clone |
| `handoff/presentation/` | The scripts that build the slides |

Not here and not in Drive: the Python environment (exact versions in `requirements_lab_vm.txt`; steps 1–3 also need RAPIDS cuML and a CUDA GPU), about 4.6 GB of embeddings that the pipeline scripts rebuild, and licensed foot-traffic data (Advan, SafeGraph), which stays on the lab server. The scraping code lives in a separate repository on Suyash's machine. Its documentation arrives with the data, in `Reviews Data/`.

Commits before November 16, 2026 also contain the result files. They were moved to Drive; the old copies in the history are out of date after labeling.

Private research data. Don't share outside the research group.
