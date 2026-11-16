# Data dictionary

What every column in the Google Drive files means. Folder names match `google_drive_upload/`. "Provisional" files end in `_PROVISIONAL` until the hand labeling is imported; their columns don't change after it.

The five dimensions appear in column names as `Availability`, `Accessibility`, `Accommodation`, `Affordability` and `Acceptability`. Below, `{dim}` stands for any one of them.

Every store has a `store_id` (called `master_store_id` in the store list). It joins every file. Census tracts join on `GEOID` (11 digits).

---

## 2_Store_List

### NC_Grocery_POIs_v2.xlsx / .csv

One row per store, 4,442 rows. Built from the frozen review database (`export_poi_table.py`).

| Column | Meaning |
|---|---|
| `master_store_id` | Store ID used everywhere else |
| `store_name`, `business_name` | Name on Google; name in the source list it came from |
| `street`, `city`, `state`, `zip`, `full_address` | Address |
| `latitude`, `longitude` | Location (WGS 84). Blank for one farm whose Google pin is outside NC |
| `category`, `type`, `subtypes` | Google's main category and other categories |
| `google_place_id`, `google_feature_id` | Google's IDs for the store page |
| `phone`, `google_rating` | As shown on Google when probed |
| `business_status` | Open status on Google |
| `target_reviews` | Google's live review count (the true count wherever a scraper saw the page) |
| `reviews_scraped` | Reviews we hold |
| `scrape_status` | `complete` (4,400) or `incomplete` (42) under the written completion rule |
| `status_reason` | Why a store was ruled the way it was, when a person or rule decided it |
| `dewey_placekey`, `dewey_brand`, `dewey_chain`, `dewey_top_category`, `dewey_naics`, `dewey_match_grade` | Match to Dewey (SafeGraph) places, 3,393 stores. Grade A/B/C = match confidence |

`NC_Grocery_POIs_v1_rows_not_in_v2.csv` lists the few stores in the October 11 version that the final version dropped.

---

## 3_Scores

### store_scores_A_teresa_method.csv and store_scores_A_frozen_raleigh_model.csv

Teresa's method. One row per store (4,442). "Frozen" uses Teresa's saved Raleigh topic model on NC reviews; the other refits the topic model on NC.

| Column | Meaning |
|---|---|
| `store_id`, `place_id`, `feature_id`, `store_name`, `name`, `city`, `county`, `category`, `lat`, `lon`, `rating`, `target_reviews` | Store details |
| `{dim}_n_reviews` | Reviews that count toward this dimension (in a valid topic matched to it) |
| `{dim}_raw_mean_signal` | Mean star signal of those reviews: 1 star = −1, 2–4 stars = 0, 5 stars = +1 |
| `{dim}_raw_se_signal` | Its standard error, floored at 0.000001 as in Teresa's code |
| `{dim}_pooled_mean_signal` | The same after partial pooling (hierarchical Bayesian model). This is the store's score |
| `{dim}_pooled_lo95`, `{dim}_pooled_hi95` | 95% credible interval |
| `{dim}_shrinkage` | Pooled minus raw |
| `{dim}_pooled_z` | The pooled score as a z-score across NC stores |

Blank means the store has no reviews on that dimension. Caution: with Teresa's error floor, a store whose few reviews are all 5-star is barely shrunk, so it can top a ranking on one or two reviews. Filter on `{dim}_n_reviews`.

### store_scores_B_improved.csv

Pipeline B. One row per store that has at least one review on a dimension (4,371).

| Column | Meaning |
|---|---|
| store details | As above |
| `{dim}_n` | Reviews that count toward this dimension. A review counts when any of its sentences is about the dimension |
| `{dim}_b_mean` | Pooled store score from text sentiment (−1 to +1 scale), with county and store-type levels |
| `{dim}_b_lo95`, `{dim}_b_hi95` | 95% credible interval |
| `{dim}_b_z` | z-score across NC stores |
| `{dim}_bstars_mean` | The same model fed star signals instead of text sentiment (a check on how much the text changes things) |

### store_local_z_B_visits.csv and store_local_z_B_equal.csv

Each store compared with the other stores its shoppers could reach in 16 minutes, not with the whole state. `{dim}` = local z-score, averaged over the tracts whose market includes the store, weighted by population. Only markets with 3 or more stores count. `_visits` weights competing stores by foot traffic; `_equal` counts them the same.

### tract_scores_A_1mile.csv and tract_scores_A_frozen_1mile.csv

Teresa's neighborhood scores. One row per NC census tract (2,672).

| Column | Meaning |
|---|---|
| `GEOID`, `county_fips`, `lat`, `lon`, `land_sqmi` | Tract ID, county, centroid, land area (2022 Census gazetteer) |
| `pop`, `black`, `income` | Population, Black population, median household income in dollars (ACS 2022 5-year) |
| `pct_black` | `black / pop`, a share from 0 to 1 |
| `dist_nearest_store_mi` | Straight-line miles from the tract centroid to the nearest study store |
| `dist_nearest_z` | That distance as a z-score |
| `E_{dim}` | Average of store `{dim}_pooled_z` scores, each weighted by exp(−miles from the tract centroid). Units: store z-scores |

### tract_scores_B_drive_time_visits.csv and tract_scores_B_drive_time_equal.csv

Neighborhood scores from real drive times on OpenStreetMap roads (free-flow, so a little optimistic). Same tract details as above, plus:

| Column | Meaning |
|---|---|
| `drive_min_nearest`, `drive_min_nearest_z` | Minutes to the nearest study store, and its z-score |
| `n_stores_16`, `n_stores_32` | Study stores within a 16- and 32-minute drive |
| `E_{dim}_16`, `E_{dim}_32` | The typical reachable store: mean of Pipeline B store scores within 16 or 32 minutes, weighted by foot traffic (`_visits`) or equally (`_equal`). Standardized across tracts, population-weighted |
| `best_{dim}_16`, `best_{dim}_32` | The best reachable store's score, standardized the same way |
| `E_{dim}_fallback` | `E_{dim}_16`, or the 32-minute value when no store is within 16 minutes |

Blank means no study store within that drive time (162 tracts at 16 minutes, 20 at 32).

### regressions/

`step7_regressions_*.csv`: Teresa's tract regression for each outcome. Columns `b_*` are coefficients, `p_*` p-values, `r2` R². Predictors: `pct_black_c` (share), `income_c` (dollars / 10,000) and their interaction, population-weighted. The `b6_local_market_*.json` and `b_pipeline.json` files hold the same for the drive-time markets and Pipeline B's run summary.

---

## 4_Review_Level

### NC_reviews_with_dimensions.parquet

One row per review, all 1,941,538. Read with `arrow::read_parquet()` in R or `pandas.read_parquet()` in Python. `..._preview.csv` is 2,000 random rows with text.

| Column | Meaning |
|---|---|
| `review_db_id` | Row ID in the master database |
| `review_id` | Google's review ID |
| `store_id`, `place_id`, `store_name`, `category`, `city`, `county`, `lat`, `lon` | The store |
| `stars` | 1 to 5. 636 reviews have none |
| `has_text`, `text` | Whether the reviewer wrote anything, and what (977,398 have text) |
| `date_relative` | Google's wording, such as "3 years ago" or "Edited a year ago" |
| `edited` | The date shown is the last edit, not the first post |
| `review_year_approx` | Scrape date minus the relative age, rounded down. Good to about a year. Blank for 980 reviews with no date |
| `scraped_at` | When we collected it |
| `has_owner_response`, `is_local_guide`, `reviewer_review_count` | As shown on Google |
| `store_scrape_status`, `store_target_reviews` | The store's completion status and Google's count |
| `store_holds_newest_only` | The store has over 1,500 reviews on Google, so we hold only its newest. True for 257 stores and a quarter of all reviews |
| `store_target_exactly_1500` | The store's count is an old capped number, so its true size is unknown (109 stores) |
| `a_topic` | Pipeline A topic (−1 = outlier, about 1.7% of text reviews) |
| `a_dim1`, `a_dim2`, `a_dim3` | Dimensions that topic matches, best first (most topics have one) |
| `a_topic_valid` | Whether the topic counts toward store scores |
| `frozen_topic`, `frozen_dim1..3`, `frozen_topic_valid` | The same under Teresa's Raleigh model |
| `b_{dim}` | Pipeline B: mean text sentiment (−1 to +1) of the review's sentences about this dimension. Blank = the review doesn't count toward it |
| `labels_status` | `provisional` or `human` |

Reviewer names are left out on purpose. They are in the database.

---

## 5_Human_Labeling

| File | Meaning |
|---|---|
| `web_labeler_page/` | The labeling page's files, to republish it from another account |
| `inputs/gold_sample_260_reviews.csv` | The 260 reviews raters label: 170 random, plus 30 each picked by keyword for parking/location, hours/payment and stock/selection |
| `inputs/topics_A_83.csv` | The 83 Pipeline A topics with keywords and example reviews |
| `inputs/topics_A_draft_groups.csv` | The draft grouping of near-duplicate topics (hand-made, not approved) |
| `inputs/topics_B_sentence_83.csv` | Pipeline B's sentence topics |
| `imported_labels/` | After labeling: per-rater files, `gold_consensus.csv`, `step5_manual_scores.csv`, `b2_manual_scores.csv`, `step5_import_meta.json`. See *FOR_CAROLINE*, Section 4 |

---

## 9_Database

| File | Meaning |
|---|---|
| `Step_5_master_reviews.db` | The dataset (SQLite). Tables: `stores` (4,442), `reviews` (1,941,538), `quarantined_reviews` (29,321 set aside, each with a reason), `accepted_flags` (validator warnings accepted by a person). Content equals freeze v2 |
| `Step_5_out_of_scope.db` | 4,736 stores ruled out of scope (gas stations, convenience stores and similar), with 105,892 reviews |
| `Step_5_closed_stores.db` | 67 stores verified closed, with 7,836 reviews |
| `Step_5_excluded_stores.db` | 221 duplicates and unfixable pages |
| `Step_1_NC_Food_Retail_Frame.db` | The 36,820-store frame the list was cut from |
| `step1_reviews.parquet` | Cleaned text of the 975,388 written reviews, an input to the pipelines (goes in `NC Pipeline/data/` of the repo) |
| `SHA256SUMS.txt` | Checksums |

The `reviews` table also has `reviewer_name` (a display name, not an ID), `helpful_count`, `photo_count` and `service_type`. The last three are empty or unreliable: Google doesn't show them on the page any more.
