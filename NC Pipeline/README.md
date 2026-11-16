# NC Pipeline: Teresa's review-topic method at North Carolina scale

This folder scales Teresa Groton's Raleigh pilot to every NC grocery store's Google reviews.
Her method has five steps:

1. Embed each review.
2. Reduce the embeddings with UMAP.
3. Cluster them into topics with HDBSCAN, using BERTopic.
4. Match each topic to the five access dimensions.
5. Human check of the topics (app).
6. Pool the reviews into store scores.
7. Census-tract scores and the equity regression.

The full explanation, every decision test and Pipeline B (improvements) are in `../PIPELINE_GUIDE.md`.

Background: `../Teresa Project/Teresa Files/Aspatial_Access_NC_Plan.md`. Her notebooks:
`../Teresa Project/Code Pipeline/`.

## Run it

```bash
cd "/home/scsapre/aspatial_food_access_research/NC Pipeline"
.venv/bin/python step1_embed.py                  # ~10 min, GPU
.venv/bin/python step2_umap.py                   # a few minutes, GPU
.venv/bin/python step3_topics.py grid            # cluster-size grid with quality metrics
.venv/bin/python step3_topics.py fit --mcs N     # the final NC topic model
.venv/bin/python step3_topics.py frozen          # Teresa's Raleigh model applied to NC
.venv/bin/python step4_match.py                  # topics -> dimensions (first-person definitions)
.venv/bin/python step4_frozen.py                 # frozen arm: her topic table on NC reviews
.venv/bin/python step5_validation_input.py       # table for the human check
.venv/bin/streamlit run apps/topic_validation_app.py --server.port 8501   # human topic check
.venv/bin/streamlit run apps/gold_labeler_app.py --server.port 8502       # human gold-set labels
.venv/bin/python step5_import_web_labels.py      # labels from the shareable web labeler (apps/web_labeler) -> rater files,
                                                 # gold_consensus.csv (decision tests) and step5_manual_scores.csv (stage 6)
.venv/bin/python step6_store_scores.py [--arm frozen] [--stores raleigh|<COUNTY> --tag ...]
.venv/bin/python step7_tracts.py [--arm frozen] [--county <COUNTY>]
.venv/bin/python export_poi_table.py            # store table (NC_Grocery_POIs_v2.xlsx) rebuilt from freeze v2
.venv/bin/python -m pytest -q tests
```

Parameters and paths live in `config.py`. Every output goes to `data/`. Logs go to `logs/`.

## Input

`../Reviews Data/Step_5_master_reviews.db`, opened read-only. Its counts equal freeze v2:
- 4,442 stores;
- 1,941,538 reviews;
- 977,398 reviews with text.

The SHA-256 of the file used is recorded in `data/step1_meta.json`.

## What each step does, and where it differs from the pilot

| Step | Same as Teresa | Different at NC scale, and why |
|---|---|---|
| 1. Clean and embed | `clean_text` copied verbatim; `all-MiniLM-L12-v2` run on the cleaned text; reviews that cleaning empties are dropped (2,010) | Her store-name stopword list gained 14 NC chain tokens (Ingles, IGA, Costco, Sam's…). Brand words that are ordinary English stay in, such as *dollar*, *fresh* and *target*. |
| 2. UMAP | 30 neighbours, 5 dimensions, `min_dist` 0.05, cosine, seed 42; a 2-D version for plots | Runs on the GPU (RAPIDS cuML, BERTopic's documented GPU path). The shared VM often has under 5 GB of free RAM, and CPU umap-learn needs about 6 GB and an hour at this size. Set `ENGINE = "cpu"` in `config.py` to use her exact libraries. |
| 3. HDBSCAN + BERTopic | `min_samples` 15, leaf selection; her vectorizer (English stop words, 1–3-grams, `min_df` 2); KeyBERT and MMR (0.3) representations; saved as safetensors with c-TF-IDF | **Cluster size is re-tuned**: her 120 was set for 33K reviews, and we have 975K. The topic keywords are built with a streaming analyzer that gives identical n-grams but never holds the outlier topic's ~10M n-grams in memory. Coherence is computed on a 100K-review sample. |

UMAP and HDBSCAN run outside BERTopic, and their outputs are handed to it ("manual BERTopic").
The 5-D coordinates and cluster labels are therefore saved, and step 3 can be re-run without
re-clustering.

## Choosing the cluster size

`step3_topics.py grid` runs HDBSCAN at several minimum cluster sizes and records, for each one:
- the number of topics;
- the outlier share;
- C_v coherence;
- topic diversity (Teresa's own metrics: 0.528 / 0.709 in Raleigh).

Two other things weigh in:
- **Manual validation in step 4.** A person scores every topic, so 1,000 topics isn't feasible.
- **Topic count.** Teresa got 59 topics, 18 of which were judged invalid.

The grid results and the choice are in `data/step3_grid.csv` and below.

| Min cluster size | Topics | Outliers | Coherence (C_v) | Diversity |
|---|---|---|---|---|
| 250 | 582 | 60% | 0.457 | 0.654 |
| 500 | 308 | 57% | 0.479 | 0.707 |
| 1,000 | 168 | 58% | 0.503 | 0.729 |
| **2,000 (chosen)** | **83** | **57%** | **0.529** | **0.764** |
| 4,000 | 44 | 55% | 0.545 | 0.734 |
| 8,000 | 23 | 58% | 0.610 | 0.678 |
| *Teresa, Raleigh (120 on 33K reviews)* | *59* | *52%* | *0.528* | *0.709* |

**Choice: 2,000** (2026-10-25):
- **Coherence** equals Teresa's (0.529 vs 0.528).
- **Diversity** is the highest in the grid.
- **Outliers** are close to hers (57% vs 52%).
- **Topic count** (83) can still be validated by hand in step 4.

Coherence keeps rising as clusters get bigger, a known bias of C_v toward fewer, broader topics, so
it isn't used alone. At 8,000 the 23 topics merge distinct complaints. Revisit at the advisor meeting
if a coarser (4,000: 44 topics) or finer (1,000: 168 topics) model suits the step-4 validation better;
every grid labelling is saved in `data/step3_grid/`, and `step3_topics.py fit --mcs N` refits in about 5 minutes.

## Outputs (data/)

| File | Contents |
|---|---|
| `step1_reviews.parquet` | One row per embedded review: DB row `id`, `store_id`, `review_id`, rating, relative date, `text`, `processed_text` |
| `step1_embeddings.npy` | 975,388 × 384 float32, row-aligned with the parquet |
| `step2_umap5.npy`, `step2_umap2.npy` | UMAP coordinates (5-D for clustering, 2-D for plots) |
| `step3_grid.csv`, `step3_grid/` | The grid: labels and membership probabilities for each cluster size |
| `step3_bertopic_model/` | The final NC topic model, saved like Teresa's `bertopic_model/` (her notebook 3 loads it unchanged) |
| `step3_topic_info.csv` | One row per topic: size, keywords, KeyBERT and MMR words, representative reviews |
| `step3_review_topics.parquet` | One row per review: `topic` (−1 = outlier), `topic_prob`, plus `frozen_topic` / `frozen_sim` from the Raleigh model |
| `step3_frozen_topic_counts.csv` | How many NC reviews fall in each of Teresa's 59 Raleigh topics |
| `parents_draft.csv` | Draft parent group for each of the 83 topics, with notes to check. Hand-made and not yet approved; it does not change the topic assignments |

## Known limits

- **Non-English reviews** are embedded as written with the English model, as in the pilot. The
  plan's multilingual arm and translation for the frozen arm are not done yet.
- **Truncation:** MiniLM reads the first 128 tokens of each review (as in the pilot); 2.3% of reviews
  are longer.
- **Big stores** hold only their newest ~1,500 reviews (see `../Reviews Data/HANDOFF.md`).

## Dashboard

**NC Review Topics**: https://claude.ai/artifact/UEWTJU8ZTfjUJCXk3Hnh8A (private until shared from its Share menu).
It has a review map, the selected topic's words, star mix and sample reviews, a searchable table of all topics, and the
cluster-size grid. Its data files are built from `data/` into `dash_data/`: topics, a map sample (up to 400 reviews per
topic and 8,000 outliers), up to 18 most-typical plus 18 random reviews per topic with no reviewer names, the grid, and a summary.

## Experiment: removing reviews with no specific words

`exp_no_empty.py` (results in `data/exp_no_empty/`; the current model is untouched) removes the 78,424 reviews (8.0%) made only of
praise, filler and stop words, then redoes UMAP and HDBSCAN with the same settings. At min cluster size 2,000 it gives 78 topics,
coherence 0.567 and diversity 0.724, against 83, 0.529 and 0.764 for the current model. The word list is hand-made, in `config.GENERIC_WORDS`.
