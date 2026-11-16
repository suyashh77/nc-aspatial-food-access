# From Google reviews to food-access scores: the pipeline, end to end

*Living document. Last updated 2026-11-15 (handoff). Changes are logged at the bottom.*

*Handoff, 2026-11-15: start with `handoff/README.md`. The three guides there (for Dr. Rachunok, for Caroline, for Claude) and `run_after_labeling.sh` supersede the "what we're doing" steps below wherever they differ.*

This guide explains how Teresa Groton's Raleigh method is being scaled to all of North Carolina, with Raleigh as the
case study. (Haywood County was dropped on 2026-11-11: Caroline's presentation focuses on Raleigh.) There are two pipelines:

- **Pipeline A** copies Teresa's method exactly. It changes only what NC's size forces, and lists each of those changes.
- **Pipeline B** keeps A's frame but tests improvements one at a time. Each one is a switch, measured against A.

Every open decision has a test here. Each test gives the question, why it matters, what we ran, the result and a suggested verdict.

---

## 0. Where we are (WWW)

**Where we are** (2026-11-15, handoff).
- **Pipeline A** (Teresa's method) runs end to end for NC and Raleigh, in both arms. Stage 6 still uses a provisional valid/invalid topic list until people do the topic check.
- **The Raleigh reproduction holds:** Teresa's model on our scrape ranks her stores like she did (ρ 0.90–0.95, D8). Her Table 5.3 partly reproduces from her own store scores (D9).
- **Pipeline B runs end to end:** sentence topics, codebook definitions, text sentiment, better pooling, 16-minute drive-time markets. Each switch is tested against A (section 3).
  - Statewide, B finds the same direction as A with larger gaps: the stores typically within reach of Blacker tracts score lower on every dimension, Affordability most of all.
- **The hand-labeling page is built** (section 6): 5 raters per review, the topic check for both pipelines, every label recorded under its rater. Suyash runs labeling on Wednesday on his page (https://claude.ai/artifact/7cCD2HedpJ97FgEmnLo2uH) and pulls the results (only a page's owner can export its labels). No labels exist yet.
- **After labeling, one command reruns everything**: `run_after_labeling.sh --csv <folder>` (or `--json`). Tested end to end on 2026-11-15 with five fake raters in a sandbox copy: all 22 steps pass in about 1.5 hours.
- Scope: NC, with Raleigh as the case study. Haywood was dropped on 2026-11-11.

**Where we're going.**
- Final store and tract scores from both pipelines on **human** labels, not provisional ones.
- A Raleigh comparison for Caroline's presentation (section 3, "Raleigh case study").
- An answer to "how many dimensions do reviews contain": two for how stores perform (quality, price). What people talk about has more, with checkout and the trip itself as sixth-dimension candidates.

**What we're doing to get there.**
1. **People:** label on the page (https://claude.ai/artifact/7cCD2HedpJ97FgEmnLo2uH; raters' instructions are in `handoff/LABELING.md`). Add raters by email as Editors; a public link only lets outsiders view.
2. Then `./run_after_labeling.sh --csv <folder of Results-tab downloads>`. It imports the labels and reruns stages 6–7 (both arms), D1–D10, B4, `b_pipeline.py`, B6, `b_raleigh.py`, the review export and the Drive folder.
3. **Outside answers:** Teresa's tract list and store set. Optional: a renewed Dewey key for per-store shopper origins (section 9).

## 1. The idea in one page

A food desert is usually drawn with distance: you live more than a mile from a store, so you have no access. But a store
next door is no help if it's too expensive, filthy, out of stock or closed when you get off work. People write about
exactly these things in Google reviews. Teresa's method treats the reviews as a huge, unplanned survey. It sorts them
into the **five dimensions of access** (Penchansky & Thomas 1981, adapted to food by Caspi et al. 2012):

| Dimension | Plain meaning |
|---|---|
| Availability | Is there enough food on the shelves? Stock, selection |
| Accessibility | Can I get there? Distance, parking, transit |
| Accommodation | Does the store fit my life? Hours, SNAP/WIC, pickup, checkout |
| Affordability | Prices and value |
| Acceptability | Does it meet my standards? Freshness, cleanliness, staff |

**How one review becomes part of a score:**

```
review text ──embed──► 384 numbers ──UMAP──► 5 numbers ──HDBSCAN──► topic ("parking")
topic centroid ──cosine vs 5 definitions──► dimension(s) of the topic ──► every review in the topic inherits them
stars ──► signal (1★ = −1, 2–4★ = 0, 5★ = +1)
store × dimension: mean signal ──partial pooling──► z-score ──distance decay──► neighbourhood score ──► regression
```

The **text decides which dimension** a review is about. The **stars decide good or bad**.

---

## 2. Pipeline A: exactly Teresa's method

Code: `NC Pipeline/` (one script per stage, settings in `config.py`).
Data: 975,388 NC reviews with text, 4,383 stores (freeze v2).

| # | Stage | Teresa | Us (NC) | Changed because of scale | Status |
|---|---|---|---|---|---|
| 1 | Clean + embed | Lowercase, strip punctuation, drop store names; `all-MiniLM-L12-v2` | Same function, verbatim | 14 NC chain names added to her stopword list | ✅ `step1_embed.py` |
| 2 | UMAP | 30 neighbours → 5-D, min_dist 0.05, cosine | Same | GPU version (cuML); the VM lacks the RAM for the CPU library | ✅ `step2_umap.py` |
| 3 | HDBSCAN + BERTopic | Min cluster 120 → 59 topics | **Min cluster 2,000 → 83 topics** | 120 on 30× more reviews gives thousands of topics; coherence matches hers (0.529 vs 0.528) | ✅ `step3_topics.py` |
| 4 | Topics → dimensions | Reload the saved model and reassign every review to its nearest topic; centroid; cosine to **first-person definitions**; softmax τ = 0.11; extra labels within δ = 0.17 | Same, verbatim (`sims_to_probs` copied) | None | ✅ `step4_match.py` |
| 5 | Human check | 0–3 score per dimension per topic, or invalid; 18 of 59 invalid | Same scoring, done **per merged group** of near-duplicate topics | 83 topics; merging halves the work (test D5) | 🟡 app ready (`apps/topic_validation_app.py`) |
| 6 | Store scores | Star signal; store × dimension mean; SE floor 1e-6; PyMC `θ ~ N(μ, τ)`, μ ~ N(0, 0.5), τ ~ HalfN(0.5); 2,000 draws × 4 chains; z of pooled means (thesis eq. 4.18) | Same, including the SE floor | Labelled with store names (her code used the reviewer's name); sampled with nutpie (same model, faster) | ✅ `step6_store_scores.py` (provisional validity) |
| 7 | Neighbourhoods | w = exp(−D / 1 mile) from the tract centroid; tract score = Σ w·z / Σ w; population-weighted regression on % Black (share) + income ($10k) + interaction, both centred; ACS 2022 5-year | Same; census files downloaded from www2.census.gov | Units weren't stated in the thesis; D9 shows these match her table | ✅ `step7_tracts.py` |

The **frozen arm** runs through the same stages with her saved Raleigh model, her topic table and her 18 invalid topics (`step4_frozen.py`, then `--arm frozen`).

### 2.2 Stage 6–7 results so far (provisional validity)

**Statewide pooling** converged for both arms on every dimension: max r-hat 1.005, minimum effective sample size above 8,000, no divergences. The model is fitted on 3,242–4,334 stores per dimension.

**The SE floor is visible in the results.**
- A store whose reviews on a dimension are all 5★ gets SE = 0, which is floored at 1e-6. The model then treats it as perfectly measured and doesn't shrink it at all.
- In a test run on one small county, a deer processor (9 reviews), a farm (12) and a nutrition shop (**3 reviews**) tied for the top Acceptability score (z = 1.41).
- Statewide, 369 stores sit at the floor for Acceptability alone.
- Pipeline A keeps this because it's Teresa's method. Fixing it is B5.

**Small samples break Teresa's model.** In that same small-county run (frozen arm, Accessibility, 21 stores) the model **did not converge** (r-hat 2.7). Its centred form can't cope when stores barely differ. Fixed in B5.

**Statewide tract regression** (NC arm, 2,624 tracts):

| Outcome | % Black | Income ($10k) | Interaction | R² |
|---|---|---|---|---|
| Distance to nearest store (z) | −1.03*** | −0.03*** | 0.03 | 0.05 |
| Availability | −0.36*** | 0.00 | −0.05** | 0.02 |
| Accessibility | −0.61*** | −0.01** | −0.03* | 0.06 |
| Accommodation | −0.37*** | 0.00 | −0.05** | 0.02 |
| Affordability | −0.24*** | −0.02*** | −0.03 | 0.01 |
| Acceptability | −0.71*** | −0.02*** | −0.08*** | 0.07 |

- Statewide, tracts with a higher Black share are *closer* to a store, the reverse of Raleigh, but they face lower-scored stores on every dimension.
- R² is small. h = 1 mile is an urban rule, and a rural tract's score is effectively just its nearest store's, which is the case for B6 (local markets).
- Provisional until the human check.

**Stage 4 output (NC, first-person definitions):**
- 59 of 83 topics match Acceptability first, 13 Affordability, 4 Accessibility, 4 Accommodation and 3 Availability.
- After the reassignment, 1.7% of reviews stay outliers (Raleigh: 4%).
- Raleigh was just as lopsided: half of her labels were Acceptability.

**Geographies run through A:**
- **Raleigh**, the reproduction check (D6).
- **All of NC**.

---

## 3. Pipeline B: improvements, one switch at a time

Code: `NC Pipeline Improved/`. It reads A's outputs and never writes to them. Each switch is scored against A on the same yardsticks:
- the gold set (section 6)
- star variance explained by topic
- store-score agreement

| Switch | Problem in A | What B does | Status | Result so far |
|---|---|---|---|---|
| **B1 Remove mood** | Topics split by mood as well as subject (rude cashiers 1.5★ vs friendly cashiers 4.6★) | Remove the "5★ minus 1★" direction from every embedding, then re-cluster | ✅ `b1_sentiment_axis.py` | Mood gone, subjects kept (see below) |
| **B2 Sentences** | One topic per review, but 79% of multi-sentence reviews cover 2+ subjects | Split into sentences, cluster sentences; a review counts toward the union of its sentences' dimensions | ✅ `b2_sentences.py` (full run) | Gold recall 0.45 → 0.70, F1 0.54 → 0.65; see below |
| **B3 Data's own dimensions** | The 5 A's are imposed top-down | Group topics into dimensions by how they rise and fall across stores (section 7) | ✅ `b3_topics.py` (rerun on B1 and B2 topics) | How stores *perform* has 2 dimensions (quality, price) in every version; what people *talk about* has more, including checkout and the trip itself |
| **B4 Better definitions** | The first-person text files "selection and variety" under Acceptability; Availability recall is about 0.2 | New first-person text written from the raters' codebook, embedded sentence by sentence; tested on 20 half-splits of the gold set | ✅ `b4_definitions.py` | F1 0.651 → 0.688; Availability recall 0.50 → 0.81 (stand-in gold, see caveat) |
| **B5 Better pooling** | The SE floor means 1-review stores aren't shrunk; one mean for dollar stores and supermarkets alike | Counts of low / neutral / high reviews (ordered logit), county and category levels, non-centred | ✅ `b5_pooling.py` | Predicts held-out reviews better on every dimension; halves A's error for stores with ≤ 5 reviews |
| **B6 Local markets** | A z-score against the whole state compares a mountain town with Charlotte; exp(−miles) is an urban rule | Real drive times on OSM roads; each tract's market = stores within 16 min (32 sensitivity); stores weighted by foot traffic (optional Dewey/Advan data, `--weights equal` drops it); person-centred local z for stores | ✅ `b6_roads.py`, `b6_drive_times.py`, `b6_local_market.py` | 489K people have no study store within 16 min; the typical reachable store is worse in Blacker tracts, the best one is better |
| **B7 Text sentiment** | "Good selection but prices are high" (3★) scores 0 for both | A sentiment reader trained on the review vectors, applied to each sentence; a dimension takes the mean of its own sentences | ✅ `b7_text_sentiment.py` | Gets gold polarity right 84% of the time (stars: 64%) |

### B1 result: removing the mood direction

| | A | B1 |
|---|---|---|
| A linear classifier can tell 1★ from 5★ reviews (balanced accuracy) | 0.95 | **0.51** (chance) |
| Share of star variance explained by which topic a review is in | 0.32 | **0.12** |
| Spread of mean stars across topics (SD) | 0.69 | 0.50 |
| Rude-cashier and friendly-cashier reviews in the same top topic? | No (topics 9 vs 22; overlap 0.35) | **Yes** (topic 86; overlap 0.85) |
| Topics / outliers after reassignment | 83 / 1.7% | 88 / 2.8% |
| Gold-set multi-label F1 (subject accuracy) | 0.60 | 0.59 |

- One direction carried essentially all the mood a linear model can read; on that direction alone it reaches 0.91.
- After removing it, topics group by **what** a review is about, and stars are left to say **how good**. That's the division of labour Teresa's design assumes.
- Subject accuracy is unchanged.
- **Verdict:** adopt in Pipeline B. Use B1 topics as the base for B3. Topic table: `out/b1_topics.csv`.

### B2 result: sentences instead of whole reviews (all 975K reviews)

The reviews split into 2.0M sentences, which cluster into 83 sentence topics (`out/b2_topics.csv`).

| | A (whole review) | B2 (sentences) |
|---|---|---|
| Reviews counting toward any dimension | 69% | **90%** |
| Dimensions per review | 0.99 | 1.74 |
| Availability / Accessibility / Accommodation share of reviews | 11% / 5% / 6% | **29% / 18% / 13%** |
| Gold precision / recall / F1 | 0.70 / 0.45 / 0.54 | 0.62 / **0.70** / **0.65** |
| Rude and friendly cashiers in the same topic? | No | Yes (overlap 0.90) |
| Store rankings vs A (ρ) | | 0.83 / 0.81 / 0.89 / 0.81 / 0.98 |

- **Catch:** my automatic "generic" filter caught only 4 sentence topics. Vague ones still receive dimensions: "ok, good, disappointed" → Acceptability, "visit, going, definitely" → Accessibility, "location, nc, area" → Acceptability. That's why precision drops.
- **Verdict:** keep B2, but its sentence topics need the same human check as A's before store scores (a second run of the topic-check app on `b2_topics.csv`).

### B3 rerun: real topics instead of hand-made groups

`b3_topics.py --base b1|b2`. Items are now the topics themselves (79 after dropping generic ones), on the 1,803 stores with ≥ 150 reviews. For B2, "how the store does" uses B7's text sentiment, which gives 17 topics enough evidence per store (B1 has 3).

| View | B1 topics | B2 sentence topics |
|---|---|---|
| Performance (how a store does) | **2**: overall quality (clean, staff, stocked; stability 1.00) and price (0.99) | **2**: overall quality (0.96) and a weaker second (0.82) |
| Salience (what people talk about) | 14 | 15 |

- **Performance is two-dimensional, in every version.** Even with sentence-level evidence and text sentiment, stores that do well on one access topic do well on nearly all of them, except price. For store *scores*, the five A's carry about two independent pieces of information.
- **Salience factors are mostly product mix** (meat, produce, deli, Mexican food, Target, beer and wine). They say what kind of store it is more than how accessible it is.
- **Salience themes the 5 A's don't name, stable across bootstraps:**
  - **checkout**: self checkout, registers, long lines (0.96 in B2);
  - **service and money**: refunds, receipts, customer service (0.92);
  - **the trip**: parking, pickup and crowding together (0.78 in B2; parking and convenience in B1);
  - **stock and shelves** (0.70).

  These are the candidates for a sixth dimension. The checkout factor is the strongest. The B4 codebook files it under Accommodation, and so do the raters' rules. Whether people treat it as its own thing is a question for the "other subject" write-ins.

### B4 result: definitions written from the codebook

Teresa's text, on the B2 sentence route, against new text that follows the raters' rules (selection → Availability; checkout, layout, pickup, hours → Accommodation). Mean over 20 random half-splits of the gold set; τ and δ re-tuned on each training half.

| Definitions | Test F1 | Availability recall | Accommodation recall | Acceptability recall |
|---|---|---|---|---|
| Teresa's first-person (A) | 0.651 | 0.50 | 0.36 | 0.95 |
| Teresa's, sentence by sentence | 0.601 | 0.54 | 0.19 | 0.90 |
| **B4 text, sentence by sentence** (chosen) | **0.688** | **0.81** | **0.57** | 0.88 |
| B4 text, whole paragraph | 0.672 | 0.88 | 0.41 | 0.82 |
| Learned from labeled reviews (prototype) | 0.659 | 0.80 | 0.62 | 0.76 |
| B4 text + prototype | 0.696 | 0.80 | 0.60 | 0.85 |

- **Caveat:** the gold set is still Claude's stand-in, coded with the same rules the new text follows. This shows the text follows the codebook, not that the codebook is right. Rerun after the human labels: `decisions/common.gold()` switches automatically.
- **Watch:** under the new text, 32 of 83 sentence topics go to Availability first, many of them product topics ("great produce section"). Whether that's selection (Availability) or quality (Acceptability) is exactly what the human gold set has to settle.
- Pipeline B uses the text-only variant, so nothing is fitted to the stand-in labels (`out/b4_definition_vectors.npz`).

### B5 result: pooling that predicts better

Each store's reviews were split in half at random. Each model was fitted on one half and predicted the other half's mean signal (stores with ≥ 3 held-out reviews; error = n-weighted mean squared error; lower is better).

| | Raw mean | A (Teresa) | B5a (SE fix only) | **B5** |
|---|---|---|---|---|
| Availability | 0.037 | 0.033 | 0.026 | **0.023** |
| Accessibility | 0.069 | 0.063 | 0.042 | **0.040** |
| Accommodation | 0.077 | 0.072 | 0.051 | **0.046** |
| Affordability | 0.025 | 0.022 | 0.018 | **0.018** |
| Acceptability | 0.012 | 0.011 | 0.010 | **0.010** |
| Stores with ≤ 5 training reviews, Availability | 0.197 | 0.175 | 0.102 | **0.089** |
| … Accessibility | 0.208 | 0.189 | 0.093 | **0.087** |

- **B5a** keeps Teresa's model but takes each store's SE from the statewide spread of signals (σ_w / √n) instead of its own few reviews. That alone does most of the work.
- **B5** models the counts of 1★ / 2–4★ / 5★ reviews directly (ordered logit), with county and store-category levels, non-centred. It's best on every dimension.
- All fits converged (r-hat ≤ 1.03, no divergences).
**Full NC fit** (`out/b5_store_scores.csv`):

| | Availability | Accessibility | Accommodation | Affordability | Acceptability |
|---|---|---|---|---|---|
| Stores at A's SE floor | 750 | 862 | 882 | 681 | 369 |
| Median reviews behind A's top-10 stores | **2** | **1** | **1** | **2.5** | **2.5** |
| Median reviews behind B5's top-10 stores | 29.5 | 33 | 13.5 | 39.5 | 123 |
| Rank agreement with A (ρ): B5 / B5a | 0.73 / 0.90 | 0.75 / 0.89 | 0.76 / 0.94 | 0.83 / 0.91 | 0.93 / 0.97 |

- **A's "best stores" are mostly stores with one or two 5★ reviews.** That's the SE floor. B5's best stores have real evidence behind them.
- Store scores converged (r-hat ≤ 1.011, no divergences). County and category effects are constrained to sum to zero, so they can't trade off with the overall level. Even so, the spread across categories for Acceptability mixes slowly (r-hat 1.25): store scores are reliable, that one spread is uncertain.
- **Verdict:** adopt B5 in Pipeline B. For the continuous B7 text signal, Pipeline B uses B5a's likelihood with B5's county and category levels.

### B6 result: 16-minute drive-time markets

**Drive times.**
- Built from OpenStreetMap: Geofabrik NC extract, 2.07M intersections, 4.65M one-way links.
- Speeds follow OSRM's default car profile; posted limits are used where tagged.
- Free-flow (no turn or signal delays), so times are slightly optimistic.
- Every one of 2,672 tracts searched to 32 minutes (`b6_drive_times.py`, 5 minutes in all).

**Who has no store within reach** (the study's 4,442 stores):

| | Tracts | People |
|---|---|---|
| No store within a 16-minute drive | 162 | 489,418 |
| No store within 32 minutes | 20 | 34,787 |

**On Pipeline A's store scores** (isolating B6; same regression as stage 7: population-weighted, % Black as a share, income in $10k):

| % Black coefficient | A: exp(−miles) | B6: visit-weighted mean, 16 min | B6: best store within 16 min |
|---|---|---|---|
| Availability | −0.36 | −0.81 | +0.76 |
| Accessibility | −0.61 | −1.19 | +0.83 |
| Accommodation | −0.37 | −0.78 | +1.13 |
| Affordability | −0.24 | −0.24 (n.s.) | +0.96 |
| Acceptability | −0.71 | −1.18 | +0.94 |
| Drive minutes to the nearest store (z) | −1.03 (straight-line miles) | −1.33 | |

- **In tracts with a higher Black share, the typical store within a 16-minute drive scores lower than in A.** The gap is about twice A's on four dimensions.
- **The best store within reach scores higher there.** Those tracts are mostly urban, with more stores, and the maximum of more stores is higher. Access to *a* good store isn't the gap; the stores people mostly use are.
- The 32-minute version weakens most effects, and Affordability flips sign. A 32-minute market in a city covers most of the metro, so it no longer describes a neighbourhood.
- **Store local z** (each store against its shoppers' other options) ranks stores much like the statewide z (ρ 0.93–0.96). It matters most for rural stores, the ones with few competitors.
- **Foot traffic is optional.** Teresa's method uses no foot-traffic data, so B6 can run without it: `--weights equal` counts every reachable store the same (outputs end in `_equal`). The findings hold, and mostly get stronger:

  | % Black coefficient, typical store within 16 min | A scores, foot traffic | A scores, equal | B scores, foot traffic | B scores, equal |
  |---|---|---|---|---|
  | Availability | −0.81 | −0.63 | −0.68 | −0.68 |
  | Accessibility | −1.19 | −1.33 | −1.69 | −1.77 |
  | Accommodation | −0.78 | −0.91 | −1.02 | −1.39 |
  | Affordability | −0.24 | −0.47 | −1.31 | −1.97 |
  | Acceptability | −1.18 | −1.61 | −1.13 | −1.64 |

- **Caveats:**
  - With foot-traffic weights on, market weights are each store's total Advan visits. Which stores *this tract's* residents use needs the per-store home block groups (the Dewey key was rejected, `grocery-foot-traffic/README.md`).
  - Stores outside NC aren't in the study, so border tracts look worse than they are.

### B7 result: sentiment from the text

- **The sentiment reader:** a 3-class logistic regression on the review vectors (low / neutral / high stars), trained on 400K reviews.
  - On 100K held-out reviews it's 74% accurate (majority class: 60%).
  - Its score correlates 0.80 with the 1–5 stars.
- Applied to each of the 2.1M sentences. A review's sentiment on a dimension is the mean over its sentences in that dimension.

| Gold-set polarity ("+" vs "−" for each labeled dimension) | Text | Stars |
|---|---|---|
| All 424 labeled pairs | **84%** | 64% (stars say "neutral" for 33%) |
| Reviews with 2–4 stars (138 pairs) | **66%** | 0% (always neutral) |

- The stars can't say anything about a 2–4★ review. That's a third of the labeled evidence. The text gets two-thirds of it right.
- Clearly mixed reviews (one dimension above +0.3, another below −0.3) are rare: 1.1%. Most of the gain is on 2–4★ reviews, not mixed ones.
- Store scores from text and from stars rank stores alike (ρ 0.82–0.94).
- **Verdict:** adopt in Pipeline B. Stage 6's star signal stays in A.

### Pipeline B end to end (`b_pipeline.py`, then `b6_local_market.py --scores b`)

All adopted switches on:
- B2 sentence topics;
- B4 definitions (sentence by sentence, τ 0.15, δ 0.10);
- the automatic generic filter (4 topics) until people check the sentence topics;
- B7 text sentiment;
- B5 pooling with county and category levels;
- B6 16-minute markets.

A star-signal variant ("B-stars") isolates B7.

**Coverage.** 877,751 reviews (90%) count toward at least one dimension, against 69% in A.

| Share of reviews counting toward | A | B |
|---|---|---|
| Availability | 11% | 48% |
| Accessibility | 5% | 15% |
| Accommodation | 6% | 36% |
| Affordability | 18% | 27% |
| Acceptability | 58% | 66% |

The Availability share is the number to watch. The B4 definitions send product topics ("great produce section") there. The human gold set decides whether that's right (see B4).

**Store rankings against A (ρ):**

| | Availability | Accessibility | Accommodation | Affordability | Acceptability |
|---|---|---|---|---|---|
| B vs A | 0.57 | 0.45 | 0.43 | 0.58 | 0.85 |
| B-stars vs A | 0.64 | 0.53 | 0.48 | 0.70 | 0.90 |
| B vs B-stars | 0.92 | 0.87 | 0.92 | 0.86 | 0.94 |

- B and A agree on Acceptability and differ a lot on the four smaller dimensions.
- Most of the difference comes from *which reviews count* (sentences, definitions) and from pooling, not from text sentiment (B vs B-stars ≥ 0.86).
- Convergence: store scores r-hat ≤ 1.011, no divergences. The hyperparameter r-hats are ≤ 1.16, and the worst is Acceptability's category spread.

**Statewide tract regression** (B scores; B6 16-minute markets; same model as stage 7). % Black coefficient:

| Outcome | A (stage 7) | B: typical store within 16 min | B: best store within 16 min |
|---|---|---|---|
| Availability | −0.36 | −0.68 | +0.92 |
| Accessibility | −0.61 | **−1.69** | +1.22 |
| Accommodation | −0.37 | −1.02 | +1.77 |
| Affordability | −0.24 | **−1.31** | +1.76 |
| Acceptability | −0.71 | −1.13 | +0.93 |
| Drive minutes to the nearest store | −1.03 (miles) | −1.33 | |

- **Pipeline B finds the same direction as A and larger gaps.** Statewide, tracts with a higher Black share are closer to a store, but the stores typically reachable in 16 minutes score lower on every dimension.
- The biggest change from A is **Affordability**: from −0.24 to −1.31 once pooling stops rewarding one-review stores and price sentences are scored by their own text.
- The best reachable store is *better* in those tracts. More stores means a higher maximum.
- R² stays small (0.01–0.08). Race and income explain little of the variation in what's reachable.

### Raleigh case study (`b_raleigh.py`, for Caroline's presentation)

**Stores** (Teresa's 42 matched stores; ρ with her own pooled scores):

| | Availability | Accessibility | Accommodation | Affordability | Acceptability |
|---|---|---|---|---|---|
| **A frozen** (her model on our scrape) | **0.90** | **0.95** | **0.93** | **0.91** | **0.95** |
| A NC refit | 0.17 | 0.40 | 0.43 | 0.59 | 0.93 |
| B | 0.25 | 0.10 | 0.03 | 0.40 | 0.88 |

**Tracts** (Wake; % Black coefficient):

| | Teresa (Table 5.3) | A, within 1 mi of her stores (63 tracts) | B, same tracts | A, all Wake (230) | B, all Wake |
|---|---|---|---|---|---|
| Availability | −0.48 | −0.16 | −0.13 | −0.36* | −0.13 |
| Accessibility | −0.24 | −0.02 | +0.24 | −0.40* | +0.10 |
| Accommodation | −0.15 | −0.92* | +0.11 | −0.38* | −0.02 |
| Affordability | **+1.64** | −0.58* | −0.10 | −0.41* | −0.05 |
| Acceptability | −0.12 | −0.46 | +0.07 | −0.94* | −0.06 |

\* p < 0.05. None of B's Wake coefficients is significant.

**What this means for a Raleigh talk:**
- **Use the frozen arm to compare with Teresa.** It reproduces her store rankings (ρ 0.90–0.95, D8). The NC refit and Pipeline B measure the four smaller dimensions differently, so they don't line up with her stores. They agree with her only on Acceptability.
- **Her Affordability result (+1.64) depends on her 49 stores.** It reproduces almost exactly from her own store scores (D9). With every Wake store in the data, it reverses (A) or vanishes (B).
- **16-minute markets don't separate Raleigh tracts.** The median Wake tract reaches 105 stores within 16 minutes, so neighbouring tracts share nearly the same market. **With 8-minute markets** (median 21 stores; 213 of 230 tracts have one), the gaps appear:

  | % Black coefficient, all Wake, 8-minute markets | A scores | B scores |
  |---|---|---|
  | Availability | −2.00* | **−2.03*** |
  | Accessibility | −0.68 | −0.94 |
  | Accommodation | −1.36* | **−1.63*** |
  | Affordability | −1.79* | −0.59 |
  | Acceptability | −1.44* | **−1.30*** |

  Without foot-traffic weights (`b_raleigh.py --weights equal`) the Wake gaps are larger and Accessibility becomes significant too (B: Availability −3.21, Accessibility −2.59, Accommodation −1.98, Acceptability −2.34, all p < 0.01; Affordability −0.07, n.s.).

  In plain terms: in Wake, a tract whose population is 10 points more Black has typical stores within an 8-minute drive that score about 0.2 SD lower on Availability, and about 0.13–0.16 SD lower on Accommodation and Acceptability. Both pipelines agree on these three. Affordability depends on the pipeline. **For Raleigh, 8-minute markets are the version to present.**
- Top stores agree across methods where it matters: Lidl leads Affordability; Harris Teeter and Publix lead Acceptability and Accommodation.

---

## 4. Decision tests

Shared setup:
- Store scores in these tests are Teresa's raw store × dimension means before pooling, for stores with ≥ 20 reviews on that dimension.
- "ρ" is Spearman rank correlation between two versions of the store scores.
- Topics drafted as general praise or no clear subject are set aside until the human check.
- Code: `NC Pipeline/decisions/`. Results: `decisions/out/*.json`.

### D1. Which definition text did Teresa use, and which is better?
- **Why it matters.** Two versions exist: first-person (thesis slide 71) and third-person (`definition_text.json` in her demo folder). The text decides every topic's dimension.
- **Test.**
  - (a) Recompute her saved Raleigh topic probabilities from her saved centroids with each text.
  - (b) Score each text on the gold set: the review's own vector against the definitions, and the topic route.
  - (c) Check against her own 9 hand-scored demo topics.
- **Results.**

  | | First-person | Third-person |
  |---|---|---|
  | Reproduces her Raleigh probabilities | **max diff 0.0016, top dimension 41/41** | max diff 0.51, 21/41 |
  | Matches her 9 human-scored topics | **9/9** | 3/9 |
  | Gold hit rate, whole paragraph (direct / topic) | **0.79 / 0.78** | 0.65 / 0.62 |
  | Gold macro-F1 (balance across the 5 dims) | 0.44 | 0.40 (**0.49** when each sentence is embedded separately and averaged) |

  - Always guessing "Acceptability" already scores 0.68, so first-person's higher hit rate comes mostly from Acceptability.
  - Every version finds Availability poorly (recall about 0.2).
  - The literature's one-line definitions (slide 9) are worst (0.51).
- **Verdict.**
  - **Pipeline A uses first-person.** It's provably what the thesis used.
  - Neither text is good on the four smaller dimensions. Pipeline B should write better definitions (B4).
  - One small, cheap gain: embed definitions sentence by sentence.

### D2. The Nov 4 memo: is one topic per review enough?
- **Why it matters.** If sentence-level labels changed store scores a lot, replicating Teresa alone would mislead.
- **Test.** 400 random stores (151,550 reviews, 322,541 sentences). Each sentence goes to its nearest topic, and the review counts toward the union of its sentences' dimensions. Compared with A on the gold set and on store scores.
- **Results.**

  | | Whole review (A) | Sentences |
  |---|---|---|
  | Reviews counting toward any dimension | 69% | 78% |
  | Dimensions per review | 0.97 | 1.42 |
  | Reviews counting toward Availability / Accessibility | 10.5% / 4.8% | 21.5% / 11.5% |
  | Gold precision / recall / F1 | 0.70 / 0.45 / 0.54 | 0.63 / **0.67** / **0.65** |
  | Store-score agreement (ρ) | | 0.85–0.87 on Availability, Accessibility, Affordability; 0.93 Accommodation; 0.99 Acceptability |

- **Verdict.**
  - Replication first, as the memo recommended. Store rankings mostly survive (ρ ≥ 0.85).
  - The sentence route roughly doubles the evidence for the small dimensions and finds a third more of the gold labels. **It belongs in Pipeline B (B2), reported as a separate extension.**

### D3. How many topics: 44, 83 or 168?
- **Test.** The full stage-4 route at each cluster size, then compare gold scores and store rankings.
- **Results.**
  - Gold F1: 0.59 at 44 topics, 0.60 at 83, 0.60 at 168.
  - Store rankings against 83 topics: ρ 0.87–0.996 for 44 topics and 0.90–0.994 for 168.
- **Verdict.** **Keep 83.** The choice barely moves the results, and 83 is half the human-check work of 168.

### D4. Reassign outliers to their nearest topic?
- **Why it matters.** At fit, 57% of reviews are outliers. Teresa's saved-model reassignment leaves 1.7%.
- **Results.**
  - Reviews with a dimension: 304K → 674K.
  - Stores with ≥ 20 Availability reviews: 529 → 1,554. Accessibility: 277 → 541.
  - Store rankings with and without reassignment: ρ 0.88–0.96.
  - Reassigned outliers are a bit less precise on the gold set (0.65 vs 0.75) with similar recall.
- **Verdict.** **Yes, reassign.** It's what Teresa did, it triples coverage, and it barely changes the ranking of stores that were already scored.

### D5. Merge near-duplicate topics before the human check?
- **Results.**
  - In 20 of the 23 draft groups, every topic shares the same top dimension. 22 of 23 very close pairs (cosine ≥ 0.85) share it too.
  - Merging cuts what a person has to score from 83 items to 41.
  - Store rankings: unchanged for Acceptability (ρ 0.998), 0.85–0.92 for the others. That shift comes from second and third labels.
- **Verdict.** **Yes, merge for the check.** Store scores keep the per-topic labels from stage 4. The app copies each group's human score to every topic in it.

### D6. Which arm feeds the final scores: Teresa's frozen Raleigh model, or a model refit on NC?
- **Results.**

  | | Frozen Raleigh model | NC refit (A) |
  |---|---|---|
  | **Raleigh reproduction**, 42 of her 49 stores, our newer scrape, ρ vs her pooled scores | **0.87, 0.90, 0.91, 0.87, 0.94** | 0.18, 0.42, 0.44, 0.54, 0.92 |
  | Gold F1 (NC reviews) | 0.50 | **0.54** |
  | Asian grocery reviews with a dimension | 55% | **74%** |

  - The two arms agree on store rankings for Acceptability (0.98) and less on the rest (0.74–0.85).
  - Stores were matched by place ID (14) and then by Google feature ID (28 more).
- **Verdict.**
  - **The method reproduces**: her model on our data recovers her store rankings. That's the trust check her plan asked for.
  - A refit model draws its own topics, so the minority dimensions get different reviews. Present both arms:
    - frozen, for comparison with Raleigh;
    - refit, for NC-specific stores (ethnic markets, dollar stores).

### D8. Does the Raleigh reproduction hold after pooling?
- **Test.** Stage 6 run for her 42 matched stores in both arms, against her thesis `store_scores.csv`.
- **Results.**

  | ρ vs her pooled store scores | Availability | Accessibility | Accommodation | Affordability | Acceptability |
  |---|---|---|---|---|---|
  | Frozen arm (her model on our scrape) | **0.90** | **0.95** | **0.93** | **0.91** | **0.95** |
  | NC refit | 0.17 | 0.40 | 0.43 | 0.59 | 0.93 |

  - Average pooled levels match too: Availability 0.54 vs her 0.56, Affordability 0.52 vs 0.53, Acceptability 0.47 vs 0.48.
  - Side finding: the `_z` column in her `store_scores.csv` is the z of the **raw** means (r = 1.00), while the thesis (eq. 4.18) says pooled. Pipeline A follows the thesis. D9 can't tell the two apart.
- **Verdict.** Her pipeline is reproduced end to end on new data. The NC refit is a different measurement of the four smaller dimensions, not a broken copy (D6).

### D9. Stage 7's unstated choices: which tracts, which units, which z?
- **Why it matters.** The thesis gives the formulas but not the units of % Black or income, or the list of Raleigh tracts.
- **Test.** Rebuild her Table 5.3 from her own 49 store scores, across tract sets, units and both z versions. Pick the combination closest to her R² values and coefficient signs.
- **Results.**
  - Best: Wake tracts within 1 mile of one of her stores (63 tracts), % Black as a share, income in $10,000s.
  - Signs agree on 78% of coefficients. Mean R² error is 0.04.
  - **Affordability matches almost exactly**: % Black 1.649 vs her 1.640; income 0.042 vs 0.041. Acceptability R² 0.41 vs 0.415.
  - Distance to the nearest store matches least (her R² 0.36, ours 0.26). She probably measured distance to all supermarkets, not just her 49.
- **Verdict.** Units settled (share, $10k). Her exact tract list is unknown, so the Raleigh regression is reproduced only approximately. Ask Teresa for her tract list and her distance-to-nearest store set.

### D10. Where do the definitions sit in the space the topics were clustered in?
- **Why it matters.** Stage 4 matches topics to definitions in the 384-D embedding space, but HDBSCAN found the topics in the 5-D UMAP space. If a definition lands outside "its" topics there, the matching is fragile.
- **Test.** Step 2 saved coordinates but not the fitted UMAP model. So each definition is placed by its 50 nearest reviews in 384-D: which topics those reviews are in, and their median 5-D position (`decisions/d10_definitions_in_umap.py`).
- **Results.**

  | Definition (Teresa's first-person) | Topics of its 50 nearest reviews | Share in topics stage 4 assigns to it |
  |---|---|---|
  | Availability | **topic 0 "grocery store" (48 of 50)** | 0.02 |
  | Accessibility | topic 0 (25), topic 7 parking (23) | 0.46 |
  | Accommodation | topic 19 hours and closures (24), topic 0 (23) | 0.48 |
  | Affordability | topic 48 prices (43) | 0.88 |
  | Acceptability | topic 13 clean + staff (34), topic 0 (11) | 0.78 |

  - The reviews nearest the Availability definition really are about selection ("I've never seen a DG that had produce and so many items…"). They sit in **topic 0**, the 58K-review generic "grocery store" topic.
  - The provisional rules mark topic 0 **invalid**, so Availability's best evidence is dropped before scoring. Topic 0 also holds half the nearest reviews for Accessibility and Accommodation.
- **Verdict.**
  - Topic 0 is the most consequential call in the human check. Its label is generic, but its content isn't.
  - This is also why the sentence route (B2) helps the small dimensions: splitting reviews pulls the selection, parking and hours sentences out of the generic blob.

### D7. Translate non-English reviews?
- **Results.** In a 60K-review sample, 99.5% of reviews with 8+ words are English. On very short reviews ("Gut", "Bueno") the detector is unreliable.
- **Verdict.** **No translation.** Flag the reviews and move on.

---

## 5. The human check (stage 5): how to run it

**Easiest: the shareable web page** (built 2026-11-12), tab "Check topics". Labeling runs on Suyash's page (https://claude.ai/artifact/7cCD2HedpJ97FgEmnLo2uH). It covers both hand-labeling tasks (this one and the gold set in section 6), so raters need nothing installed. Details are in section 6. The Streamlit app below still works.

```bash
cd "/home/scsapre/aspatial_food_access_research/NC Pipeline"
.venv/bin/python step5_validation_input.py                       # once: builds the topic table
.venv/bin/streamlit run apps/topic_validation_app.py --server.port 8501
# on your laptop: ssh -L 8501:localhost:8501 <vm>, then open http://localhost:8501
```

The app has four tabs:
1. **Instructions.**
2. **Merge topics.** Topics start in the draft groups from `parents_draft.csv`. "General praise" and "No clear subject" groups start as invalid / too_generic. You can move a topic, mark it valid, or start a new group, then confirm each group.
3. **Score groups.** Teresa's blinded 0–3 scoring, one merged group at a time. Model guesses are hidden.
4. **Export.** Writes `data/step5_manual_scores.csv` in Teresa's format, which stage 6 reads.

Budget about 30 min for merging and 2 min per group. A second person can score a copy for an agreement statistic.

---

## 6. The gold set (our yardstick): labeling app and limits

### The shareable web labeler (recommended)

- **Link:** https://claude.ai/artifact/7cCD2HedpJ97FgEmnLo2uH. Suyash owns it and pulls the results.
- **Who can save labels:** people who open a public link from outside the organization can only view. On a personal plan, add each rater by email from the Share menu as **Editor**. Editor-level access only works while the public link is off. Raters see only their own work in the page. Only the owner sees the Results tab.
- **5 raters per review:** the page gives each rater the next review, in a shared order, that has fewer than 5 raters. Reviews fill up in order. Raters see how many people have labeled a review (`claims/` in the page's database), never their answers. The owner gets Fleiss' kappa across the 5 raters and a majority-vote consensus export (`gold_labels_consensus.csv`: a dimension counts if more than half of raters mark it).
- **Who labeled what:** every label is stored under the account that saved it. The exports carry `rater` (initials), `rater_name` and `rater_id` columns, and the Results tab warns when two people use the same initials.
- **Two tasks in one page:** "Label reviews" (the 260 gold reviews, same shuffled order and same codebook as the Streamlit app) and "Check topics" (the 83 topics in their draft groups: mark topics invalid, move topics between groups, score each group 0–3).
- **Saving:** every answer saves to the rater's claude.ai account straight away. Raters can stop and come back, and they never see each other's work.
- **For the owner, a Results tab:** progress per rater, Cohen's kappa per dimension (reviews), agreement on topic scores, review disagreements, "other subject" write-ins, and CSV downloads. The review labels match the `decisions/labels/<name>.csv` columns. The topic scores use Teresa's `step5_manual_scores.csv` format, which stage 6 reads.
- **Code:** `NC Pipeline/apps/web_labeler/` (`labeler.html`; `build_data.py` regenerates `data.js` from the same inputs). Claude can also pull everyone's labels directly from the page's database.

### The hand labeler

```bash
cd "/home/scsapre/aspatial_food_access_research/NC Pipeline"
.venv/bin/streamlit run apps/gold_labeler_app.py --server.port 8502
# laptop: ssh -L 8502:localhost:8502 <vm>  ->  http://localhost:8502
```

- **What a rater does:**
  - Types their name.
  - Reads one review at a time. Stars, store and topic are hidden.
  - Marks each dimension *not discussed / positive / negative / mixed*, or ticks *No access content*.
- **"Other subject" box:** for anything important that fits none of the five. These write-ins feed the 6th-dimension search (section 7).
- **Saving:** after every review, to `decisions/labels/<name>.csv`. Every rater sees the same shuffled order.
- **Codebook:** in the sidebar, with the judgement calls spelled out (selection → Availability; checkout and pickup → Accommodation; and so on).
- **Agreement tab:** Cohen's kappa per dimension for any two raters (Claude's stand-in counts as one), a list of disagreements to discuss, and a tally of the "other subject" write-ins.
- **Time:** about 2–3 hours for 260 reviews. Once two people have labelled, re-run the decision tests on their labels in place of Claude's.

### The current gold set

- **What it is.** 260 NC reviews: 170 random, plus 30 each chosen by keyword for parking/location, hours/payment and stock/selection, so the small dimensions have examples.
- **How it was labelled.**
  - Each review is labelled with every dimension it discusses, plus a polarity, using the thesis slide-9 literature definitions.
  - 26 reviews have no access content.
  - Files: `decisions/out/gold_sample.csv` and `gold_labels_part*.txt`.
- **Caveat.** **Claude labelled it, blind to stars and topic, as a stand-in until people do.** Two coding choices matter:
  - "Selection/variety" was coded Availability. Teresa's first-person text files it under Acceptability.
  - "Layout, checkout and pickup" was coded Accommodation.

  Two human raters on the same reviews would turn the gold set into a real measurement. That's the plan's Phase 3.

---

## 7. Letting the data name the dimensions (the "6th dimension" question)

**Idea.** Clustering grouped *reviews into topics* nobody named in advance. Do the same one level up and group *topics into
dimensions* by how they behave across stores. Psychologists found the "Big Five" personality traits this way: rate
many people on many adjectives, see which adjectives rise and fall together, and let factor analysis say how many
underlying traits there are. Here a store plays the person and a topic group plays the adjective.

**Two views, because "dimension" means two things:**
- **Salience**: what people *talk about* at a store, as each group's share of the store's reviews.
- **Performance**: how the store *does* on it, as the mean star signal within each group.

**Steps** (`NC Pipeline Improved/b3_discover_dimensions.py`):
1. Regress out store category, so a factor can't just mean "dollar store vs supermarket".
2. **How many dimensions?** Parallel analysis: keep a factor only if it beats the same data with each column shuffled.
3. Factor analysis with that many factors.
4. Re-fit on 50 bootstrap samples of stores, to keep only factors that come back every time.
5. Compare each factor's direction with the 5 definitions. **A stable factor that matches no definition is a candidate 6th dimension.**

**First pass** (1,395 stores with ≥ 150 reviews; the 23 draft topic groups):

| View | Dimensions the data supports | What they are (stability 0–1) |
|---|---|---|
| Salience | **6** | 1 Clean + staff (0.99) → Acceptability · 2 Prices (0.99) → Affordability · 3 **Pickup + parking + hours + checkout** (0.93): matches no definition well (best cosine 0.55) · 4 **Stock + crowding + quick trips** vs deli/bakery (0.95): no good match (0.54) · 5 **Cultural and international food** (0.91) · 6 Meat / butcher (0.93) |
| Performance | **2** | 1 Overall quality: clean, staff, stock, produce (1.00) → Acceptability · 2 Price (0.99) → Affordability |

**What this says so far:**
- **What people talk about has more structure than the 5 A's.** Two strong candidates:
  - **"The trip itself"**: pickup, parking, hours and checkout move together. In reviews this is one theme that the 5 A's split between Accessibility and Accommodation.
  - **"Cultural fit"**: international and ethnic food is its own theme.
- **How stores perform collapses to two dimensions, quality and price.** The other dimensions don't have enough reviews per store to show their own factor yet. Only 7 groups had enough reviews at 60% of stores. This is the strongest argument for B2: sentences give more evidence per store.
- **Caveats:**
  - The groups are my hand-made draft. The final run should use topics or the human-check groups.
  - Some salience factors may still reflect store type beyond Google's categories.

**Rerun done (2026-11-12):** see "B3 rerun" in section 3. Performance stays two-dimensional; checkout, service and money, and the trip itself are the sixth-dimension candidates.

**Next for B3:**
1. ~~Rerun on the B1 (mood removed) and B2 (sentence) topics, using topics, not my groups.~~ Done.
2. Name each candidate factor from its top reviews (section 8).
3. Write a definition for any candidate that survives, add it as a sixth, and check whether it takes topics that matched badly without taking topics from the other five.

---

## 8. From vectors back to words

- A topic's 384-number centroid is just the average of its reviews' vectors. Those are saved in row order, so there's never a need to map 5-D UMAP points back up.
- To put words on any vector (a centroid, a factor):
  1. The **reviews closest** to it.
  2. Its **c-TF-IDF keywords**.
  3. For a factor: the reviews at its two extremes.
  4. An LLM writes a one-line name from those, and we embed the name to check it lands back near the vector.
- True inversion (vec2text) has no ready model for MiniLM. It would mean training one, a multi-week side project.

---

## 9. Open items

**Needs people:**
- [ ] **Topic check** (stage 5), **gold labels** (5 raters per review) and the optional **sentence-topic check**, on the labeling page, Wednesday. Topic 0 is the most important call (D10).
- [ ] Then `./run_after_labeling.sh --csv <folder>`, which does every rerun. Afterwards replace the provisional numbers in this guide.

**Needs outside answers:**
- [ ] Ask Teresa for (a) her Raleigh tract list, (b) the store set behind "distance to nearest store", and (c) whether the thesis used raw-mean or pooled z (her data file and her text disagree).
- [ ] *Optional (Teresa's method uses no foot-traffic data):* renew the Dewey API key, then pull per-store visitor home block groups (`grocery-foot-traffic/README.md`, "Not done yet"). B6 would then weight each tract's market by where *its* residents shop, not by stores' total visits.
- [ ] Copy `NC Pipeline/data/exports/NC_Grocery_POIs_v2.xlsx` to the scraping repo's `Data/exports/` (on the scraping machine) and tick item 2 of `Reviews Data/HANDOFF.md`.
- [ ] Zip the scraping repo (without `Auth/` and `Data/`) from Suyash's PC and upload it to Drive.

**Analysis, can run any time:**
- [ ] B6 counts only the study's NC stores, so border tracts look worse than they are. Adding stores across the state line would fix it.
- [ ] B6's 32-minute market flips the Affordability result. Decide whether 32 minutes stays as the sensitivity case or becomes a rural-only fallback (`E_*_fallback` already does that).

**Done at handoff (2026-11-15):**
- [x] Stage 7 for the frozen arm (`step7_tracts_frozen.csv`): same direction as the NC arm, smaller on Availability (−0.12) and Affordability (−0.14).
- [x] The analysis export the scraping handoff asked for: one row per review with every pipeline's labels and the censoring flags (`export_review_table.py` → `data/exports/NC_reviews_with_dimensions.parquet`).
- [x] Label import from the Results-tab CSVs (`--csv-dir`), identical to the JSON route; tests added.
- [x] `run_after_labeling.sh` and `build_drive_package.py`; handoff docs in `handoff/`.

**Done since 2026-11-10:**
- [x] B5 pooling fix (holdout-tested); B6 local markets with real drive times; B3 rerun on B1/B2 topics; B4 definitions; B7 text sentiment; Pipeline B end to end.
- [x] Store table rebuilt from freeze v2 (`export_poi_table.py`, with a test).
- [x] The 636 reviews with no star rating: Pipeline A already does exactly what Teresa's notebook 6 does. They count as "Neutral" in the counts and the SE but are left out of the mean, because their signal is missing. Nothing to fix in A. B drops them.
- [x] Haywood dropped from scope (2026-11-11).

## Changelog

- **2026-11-10**:
  - Created.
  - Pipeline A stage 4 built (`step4_match.py`, first-person definitions).
  - Decision tests D1–D7.
  - Topic-check app (`apps/topic_validation_app.py`).
  - Gold set (260 reviews).
  - Pipeline B folder with B1 (running), B2 (ready) and B3 (first pass).
  - Later the same day: B1 finished (mood removed, subjects kept). Full B2 run started.
  - Later still:
    - Pipeline A stages 6 (`step6_store_scores.py`) and 7 (`step7_tracts.py`) built, plus the frozen arm (`step4_frozen.py`).
    - Run for NC, Raleigh and Haywood. D8 (Raleigh reproduction through pooling) and D9 (Table 5.3 units) added.
    - Full B2 results added. Hand labeler (`apps/gold_labeler_app.py`) built.
    - ACS 2022 tables and the tract gazetteer downloaded to `NC Pipeline/data_external/census/`.
- **2026-11-12**: Shareable web labeler for both hand-labeling tasks (topic check and gold set): per-rater saving, blind between raters, agreement and CSV export for the owner (`apps/web_labeler/`).
  - Later the same day: each review now gets 5 raters (handed out in a shared order); rater name and account id in every export; Fleiss' kappa and majority-vote consensus export; Results tab is owner-only.
- **2026-11-12 (later)**:
  - Label import (`step5_import_web_labels.py`). Decision tests and stage 6 use human labels automatically when present.
  - Web labeler: Task 3, the sentence-topic check.
  - B3 rerun, B4, B5 (holdout + full fit), B6 (OSM drive times, Advan weights), B7, and Pipeline B end to end (`b_pipeline.py`).
  - Raleigh case study (`b_raleigh.py`) and D10 (definitions in UMAP space).
  - Store table rebuilt from freeze v2. No-star item closed. Haywood dropped. WWW and open items rewritten.
  - Raleigh with 8-minute markets added (`b_raleigh.py`).
  - Foot-traffic weights in B6 made optional (`--weights visits|equal`; Teresa's method uses none). Results hold without them.
- **2026-11-15 (handoff)**:
  - Stage 7 frozen arm run. Review-level export (`export_review_table.py`). Pipeline B's review labels moved into `b_review_dims.py` (shared by `b_pipeline.py` and the export; reproduces its counts exactly) and now saved as `out/b_review_dims.parquet`.
  - Label import accepts the Results-tab CSVs (`--csv-dir`). `run_after_labeling.sh` reruns everything after labeling; tested end to end in a sandbox with five fake raters. `build_drive_package.py` builds the Drive folder.
  - Handoff docs (`handoff/`): guides for Dr. Rachunok, Caroline and Claude, data dictionary, file manifest, final presentation. Caroline republishes the labeling page and runs labeling. The Nov 13 draft README moved to `_superseded/`.
- **2026-11-16 (storage split, no method change)**:
  - GitHub now holds the code only. Result files, working files, documents and the slides moved to the Google Drive folder (`1_Docs_and_Reports`, `2_Presentation`, `3_Datasets`), built by `build_drive_package.py`. `place_data.sh` puts the Drive data into a fresh clone; it replaces `fetch_data.sh`. No pipeline code or number changed.
- **2026-11-16 (labeling plan, no method change)**:
  - Suyash runs the labeling on his own page (https://claude.ai/artifact/7cCD2HedpJ97FgEmnLo2uH) instead of Caroline republishing it, then runs `run_after_labeling.sh` and hands the final Drive folder to Caroline. `handoff/FOR_CLAUDE.md` has the steps for when labeling is done.
- **2026-11-16 (Drive layout and docs, no method change)**:
  - The Drive folder is now `README.docx`, `docs/`, `presentation/`, `labeling/`, `data/` (was `3_Datasets`), `results/`, `code/`. `build_drive_package.py` builds that layout; `place_data.sh` takes the `data/` folder.
  - The people-facing docs (`handoff/HANDOVER.md`, `LABELING.md`, `USING_THE_DATA.md`, `DATA_DICTIONARY.md`) replace `FOR_DR_RACHUNOK.md` and `FOR_CAROLINE.md`, are tracked in git, and are built into Word by `handoff/md2docx.py`.
