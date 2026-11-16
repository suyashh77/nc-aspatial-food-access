# Using the data

For anyone using the scores in a study. Column meanings are in `DATA_DICTIONARY.docx`.

## Which file

| Question | File |
|---|---|
| Which grocery stores exist in NC? | `data/store_list/NC_Grocery_POIs_v2.xlsx` |
| How does a store score? | `data/scores/store_scores_B_improved*.csv` (recommended) or `store_scores_A_teresa_method*.csv` |
| How good are the stores a neighborhood can reach? | `data/scores/tract_scores_B_drive_time_*.csv` (16-min drive) or `tract_scores_A_1mile*.csv` (Teresa's 1-mile decay) |
| What did people write? | `data/review_level/NC_reviews_with_dimensions*.parquet` (`_preview.csv` for a first look) |
| Raleigh vs Teresa | `results/Raleigh_case_study*.xlsx`, files behind it in `data/raleigh_case_study/` |

- Store IDs join everything (`store_id`, `master_store_id` in the store list). Tracts join on 11-digit `GEOID`.
- Review file is Parquet because CSV would be over 1 GB. R: `arrow::read_parquet()`. Python: `pandas.read_parquet()`.

## A or B

- **A** to compare with Teresa. The "frozen" version runs her own Raleigh model on our reviews and matches her store rankings (rank correlation 0.90–0.95).
- **B** for anything new. Reads sentence by sentence, takes sentiment from text not stars, stops one-review stores from topping rankings.
- Say which one you used. For headline results show both.

## Traps

- **Few reviews on small dimensions.** Median store: 60 on Acceptability, 5 on Accommodation, 7 on Accessibility. Use the `_n` / `_n_reviews` columns. For store-level claims require 20+ reviews on that dimension, or use the 95% interval columns.
- **A's "best" stores are often stores with one or two 5★ reviews.** Filter on review counts if using A rankings.
- **Dollar stores are a quarter of the list** (1,001 of 4,442), on purpose. Split them out with `category` when it matters.
- **Big stores hold only their newest reviews** (257 stores, a quarter of all reviews). Flag: `store_holds_newest_only`.
- **Dates are rough.** `review_year_approx` is good to about a year. No monthly trends.
- **Drop the known bad rows:** store IDs 5379, 9368, 8852 (out-of-state addresses) and the 37 stores with a restaurant as main category.
- **Border tracts look worse than they are.** Only NC stores are in the data.
- **Rural gaps are real, not low scores.** 162 tracts (489,418 people) have no store within 16 min. Their scores are blank or come from the 32-min `_fallback` columns.

## Reading a regression coefficient

- Tract score regressed on % Black (share, 0–1) and median income ($10K units), population-weighted.
- Scores are standardized, so coefficients are in standard deviations.
- Units differ: in A's 1-mile version the unit is the spread across stores; in the drive-time versions it's the spread across tracts. Say which.
- Example: −2.0 means a tract 10 points more Black scores ~0.2 SD lower, holding income fixed.

## For the Raleigh talk (provisional numbers)

- Use the frozen A for the replication, 8-min drive-time markets for equity.
- Replication, rank correlation with Teresa on 42 matched stores:

| Availability | Accessibility | Accommodation | Affordability | Acceptability |
|---|---|---|---|---|
| 0.90 | 0.95 | 0.93 | 0.91 | 0.95 |

- A model refit on all of NC agrees with her only on Acceptability (0.93). Worth one sentence: it's a finding about the method.
- Her Affordability result (+1.64): rebuilt from her 49 stores (+1.65). With every Wake store it reverses in A (−0.41) and vanishes in B (−0.05, n.s.). Fair framing: it depends on which 49 stores were in the study. Approximate reproduction, not proof she was wrong.
- Why 8 min: 16 min from the median Wake tract reaches 105 stores, almost the whole metro. 8 min reaches 21.

| Wake, 8-min markets | A | B |
|---|---|---|
| Availability | −2.00* | −2.03* |
| Accessibility | −0.68 | −0.94 |
| Accommodation | −1.36* | −1.63* |
| Affordability | −1.79* | −0.59 |
| Acceptability | −1.44* | −1.30* |

\* p < 0.05. Coefficient on % Black (share), 211 tracts, stores weighted by foot traffic.

- Both methods agree on Availability, Accommodation, Acceptability. Affordability depends on method, so don't lead with it. Without foot-traffic weights the gaps get larger and Accessibility becomes significant.
- Recognizable stores: Lidl leads Affordability in every version. Harris Teeter and Publix lead Acceptability and Accommodation.
