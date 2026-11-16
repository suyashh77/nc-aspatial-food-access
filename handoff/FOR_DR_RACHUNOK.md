# NC aspatial food access: project report and handoff

For Dr. Rachunok. Suyash Sapre, November 15, 2026.

This report covers what we built, how the data was collected, what the results say so far, and what is wrong with the data. It also lists what is left and the decisions that are yours to make.

Short version: the statewide dataset is finished and frozen. Teresa Groton's method runs on all of North Carolina, and it reproduces her Raleigh store rankings. An improved version runs next to it. Every score is provisional until people do the hand labeling, which is scheduled for Wednesday. After that, one script produces the final numbers.

Caroline's guide (*FOR_CAROLINE*) covers the labeling session and how to use the scores. The technical handoff for whoever touches the code next is *FOR_CLAUDE*.

---

## 1. The question

Food access is usually measured by distance: if you live more than a mile from a supermarket (ten in rural areas), you live in a food desert. But a store down the street does little for you if it's too expensive, dirty, poorly stocked, or closed when you get off work.

Teresa's thesis treated Google reviews as a large, unplanned survey of what shopping at a store is like. She sorted reviews into the five dimensions of access from Penchansky and Thomas (1981), as adapted to food by Caspi et al. (2012): Availability, Accessibility, Accommodation, Affordability and Acceptability. She did it for 49 Raleigh stores and about 25,000 reviews.

This project scales her method to every grocery store in North Carolina.

---

## 2. What we did, step by step

### Step 1. Build the list of grocery stores

No single list of NC grocery stores is complete. The USDA SNAP retailer list has reliable addresses but only stores that take SNAP. Overture Maps has clean names but stale records. OpenStreetMap catches small ethnic and rural stores that commercial lists miss, but its tags are inconsistent.

So we combined them. In September we rebuilt the list from four sources: USDA SNAP, Overture Maps, OpenStreetMap, and project data (Dewey/SafeGraph places, Data Axle). Our matching tool, geomatch, linked records across sources by distance and name similarity. 91,461 records became a frame of 36,820 food retailers.

We cut that to established, bricks-and-mortar food stores (16,751 candidates) and looked each one up on Google Maps with an automated probe. The probe recorded Google's place ID, category and live review count.

The first version of the list, built in the spring from Overture alone, kept between 14% and 38% of candidates depending on the area. It was worst in rural eastern NC. It would have shown food deserts that were really holes in the list. That is why we rebuilt it.

### Step 2. Decide what counts as a grocery store

We ruled on about 300 Google categories by hand (`category_ruleset.csv` in the scraping code) and checked edge cases by hand in several manual rounds.

- **In:** supermarkets and grocery chains, discount grocers and dollar stores, independent and ethnic grocers, butchers and seafood markets, co-ops. Any chain with more than five NC locations was in automatically.
- **Out:** gas stations and convenience stores, farmers' markets and produce stands, restaurants that sell some groceries, closed stores.

Nothing was deleted. Out-of-scope stores (4,736), closed stores (67) and duplicates (221) each sit in their own database with their reviews and a written reason.

The final list has **4,442 stores** in all 100 counties. The largest categories are grocery stores (1,897), dollar stores (1,001), supermarkets (212) and department stores (174, mostly Walmart and Target).

### Step 3. Scrape the reviews

A scraper we wrote (Python and Playwright driving a headless browser) collected every review it could reach for each store. It ran between early 2026 and October 24, 2026, mostly on NC State Linux machines, one scraper per machine to stay under Google's rate limits. ISE 408 students ran some early batches.

The scraper opens each store by its exact Google place ID and refuses a page whose ID differs. The first version searched by name, and it scraped the wrong store more than once (a Sam's Club in Hendersonville resolved to one in Raleigh with 1,062 reviews). A September audit found that problem and others, and we repaired the data and the code before collecting the rest.

Result: **1,941,538 reviews**, of which **977,398 have text**. Data collection closed on October 24 and the database was frozen (freeze v2).

### Step 4. Is it authoritative?

We wanted to be able to say this is every review of every grocery store in NC. We got close, and where we didn't, the gaps are written down.

- Against Google's live review counts, capped at 1,500 per store, we hold 99.6% of reviews. 4,400 stores are complete under a written rule (a store may be missing at most 10, 20, 50 or 100 reviews depending on its size). 42 stores are each 21 to 25 reviews short after repeated full passes. Google lists those reviews but won't show them.
- Against Google's true counts, with no cap, we hold 87.9%. The difference is big stores (Section 4).
- 29,321 reviews were set aside (`quarantined_reviews`) because they belonged to another store or a scraping collision. Each has a reason. None were deleted.

What we did not do: check the final store list against Census business counts by county. The comparison file exists in the scraping code (`Frame/reports/census_validation.csv`). It's a half-day task and the best remaining test of the list.

### Step 5. Replicate Teresa on Raleigh

Before scaling anything, we checked that our data reproduces her results. We matched 42 of her 49 stores to ours and ran her saved model on our newer reviews.

Her model on our reviews ranks her stores as she did: rank correlation 0.90 to 0.95 across the five dimensions. Average scores match too (Availability 0.54 vs her 0.56, Affordability 0.52 vs 0.53).

We also rebuilt her tract regression (Table 5.3) from her own store scores. The thesis doesn't state the units or the tract list, so we tried the reasonable options. Using Wake tracts within a mile of her stores, % Black as a share, and income in $10,000s, her Affordability coefficient comes back almost exactly (+1.65 vs her +1.64), and 78% of coefficient signs match. Her tract list is still unknown, so this part is an approximate reproduction.

### Step 6. Run her method on all of NC (Pipeline A)

Pipeline A is Teresa's method, changed only where NC's size forces it. Each change is documented.

1. Clean and embed each review with the same sentence model (all-MiniLM-L12-v2). We added 14 NC chain names to her list of store names to strip.
2. Reduce to five dimensions with UMAP, same settings. It runs on a GPU because the CPU library needs more memory than the VM has.
3. Cluster into topics with HDBSCAN and BERTopic. Her minimum cluster size of 120 would give thousands of topics on 30 times more reviews. A grid search chose 2,000, which gives 83 topics with the same coherence as hers (0.529 vs 0.528).
4. Match each topic to the dimension whose definition it is closest to. We tested both versions of her definition text. Only the first-person version reproduces her saved Raleigh numbers (to within 0.002), so that is what A uses.
5. Hand-check the topics (the step waiting on Wednesday).
6. Turn reviews into store scores. Each review's star rating becomes a signal (1 star = −1, 2 to 4 = 0, 5 = +1). Each store's mean signal per dimension is pooled with a hierarchical Bayesian model, as in her notebook.
7. Turn store scores into tract scores (each tract averages nearby stores, weighted by exp(−miles)), then regress tract scores on % Black and income.

It runs end to end in two versions: one with a topic model refit on NC, one ("frozen") with Teresa's Raleigh model applied to NC reviews.

### Step 7. Improve it (Pipeline B)

Testing A turned up specific problems. Pipeline B fixes them one at a time, and each fix is scored against A on the same yardsticks.

| Problem in A | Fix in B | Result |
|---|---|---|
| One topic per review, but 79% of multi-sentence reviews cover 2+ subjects | Split reviews into sentences; a review counts toward every dimension its sentences touch | Finds 70% of the answer key's labels instead of 45% |
| The definition text puts selection under Acceptability, so Availability is rarely found | New definitions written from the raters' rules | Availability recall 0.50 → 0.81 |
| A 3-star "good selection, high prices" review scores 0 on both | Read good or bad from the text of each sentence | Right on 84% of labeled cases (stars: 64%) |
| Stores with one 5-star review aren't shrunk, so they top the rankings | Better pooling: count model, county and store-type levels | Halves prediction error for stores with ≤5 reviews |
| exp(−miles) is an urban rule, and a statewide z-score compares Boone with Charlotte | Real drive times on OpenStreetMap roads; each tract's market is the stores within 16 minutes | Shows who has no store within reach |

Two caveats. The accuracy numbers use Claude's stand-in answer key, which was coded with the same rules as B's new definitions, so they flatter B until people relabel. And the drive times ignore traffic and turn delays, so they run a little optimistic.

---

## 3. What the results say so far

All of this is provisional until the hand labeling is done.

**Statewide, distance and quality point in opposite directions.** Tracts with a higher Black share are *closer* to a grocery store than other tracts, the reverse of Teresa's Raleigh finding. But the stores they can reach score lower on every dimension. Pipeline B makes the gaps larger:

| Coefficient on % Black (share) | A: stores within ~1 mile | B: typical store within a 16-minute drive |
|---|---|---|
| Availability | −0.36 | −0.68 |
| Accessibility | −0.61 | −1.69 |
| Accommodation | −0.37 | −1.02 |
| Affordability | −0.24 | −1.31 |
| Acceptability | −0.71 | −1.13 |
| Distance or drive time to nearest store | −1.03 | −1.33 |

All A coefficients are significant at p < 0.001. Population-weighted, controlling for income. R² stays small (0.01 to 0.08): race and income explain little of what is reachable.

The best store within reach is *better* in those tracts, because they are mostly urban and have more stores to choose from. Access to *a* good store isn't the gap. The typical store people use is.

**489,418 people (162 tracts) have no study store within a 16-minute drive.** 34,787 have none within 32 minutes.

**In Raleigh, 8-minute markets show gaps that 16-minute markets hide.** A 16-minute drive covers most of the metro. With 8-minute markets, both pipelines agree that tracts with a higher Black share reach stores that score lower on Availability, Accommodation and Acceptability (about 0.13 to 0.2 SD for a 10-point difference). Teresa's positive Affordability result reverses or vanishes once every Wake store is included.

**How many dimensions are there?** We grouped topics by how they rise and fall across stores, the way the Big Five personality traits were found. How stores *perform* collapses to two dimensions: overall quality and price. What people *talk about* has more structure, and the strongest theme the five A's don't name is checkout (self-checkout, registers, long lines). Whether that is a sixth dimension is a question the raters' "other subject" notes can help answer.

The full tables, every decision test and every number above are in `PIPELINE_GUIDE.md`.

---

## 4. What is wrong with the data

Anyone using this data should know these. They are listed worst first.

**1. The answer key isn't human yet.** Every accuracy number (how often a pipeline labels a review correctly) uses 260 reviews that Claude labeled as a stand-in. Those numbers measure agreement with Claude, not with people. The topic validity list is also a draft. Both are fixed by Wednesday's labeling.

**2. Small samples on four of the five dimensions.** Most reviews are about Acceptability (staff, cleanliness, freshness). The median store has 60 reviews about Acceptability but only 5 about Accommodation and 7 about Accessibility. About 1,500 stores have five or fewer reviews on Accessibility. Store-level claims on those dimensions need the interval columns or a minimum count.

**3. Teresa's pooling rewards tiny samples.** A store whose reviews on a dimension are all 5-star gets a standard error of zero, which her code floors at one in a million. The model then treats that store as perfectly measured and doesn't shrink it. The median number of reviews behind A's top-10 stores is 1 to 2.5, depending on the dimension. A keeps this because it's her method. B fixes it.

**4. Big stores hold only their newest reviews.** 257 stores have more than 1,500 reviews on Google. Scrolling past that point takes 6 to 17 hours per store, and Google often stops serving older reviews. For those stores we hold the newest ones, between 1,500 and about 5,600. They hold a quarter of all reviews. Another 109 stores have a target stuck at exactly 1,500 from an older capped count, so their true size is unknown. Both are flagged.

**5. Dates are vague.** Google shows "3 years ago" or "Edited a year ago", not a date. We estimate the year from the scrape date, which is good to about a year. Most reviews are 2018 to 2025. Monthly or seasonal analysis isn't possible.

**6. Reviewers aren't identified.** Google shows a display name, not an ID. One "Chris" reviews 240 stores. Reviewer-level analysis (fixed effects, one person reviewing many stores) can't be done.

**7. Who writes reviews isn't who shops.** People who post Google reviews are not a random sample of shoppers. A store in a neighborhood where few people post has fewer reviews and wider intervals, and its scores reflect whoever does post. We have not measured how large this bias is.

**8. Rural stores may still be missing.** Small rural general stores and tiendas are the stores most often absent from commercial maps, and the four-source frame reduces this but can't prove it's solved. The Census check in Step 4 is the test.

**9. What counts as a grocery store is a judgment.** Dollar stores are a quarter of the list on purpose: in much of rural NC they are where people buy food. The list also includes some categories a reader might not expect, such as 70 vitamin and supplement stores (36 of them GNC), 56 health food stores and two Kohl's. Averages over "all stores" mix these together. The `category` column lets a study choose.

   A few entries are plain errors, found while drawing the maps for the final presentation. Three stores have addresses outside NC even though their state column says NC: an olive oil shop in Blairsville, Georgia, a food processing company in Norcross, Georgia, and a Hooters with a South Carolina address. And 37 stores have a restaurant as their main Google category. Some are real market-and-grill combinations; others aren't grocery stores. They're in the frozen data, so they stay flagged rather than removed. A study should drop them (`store_id` 5379, 9368, 8852, and the restaurant categories) or check them by hand.

**10. State lines.** Only NC stores are in the data. A tract near Virginia, Tennessee, Georgia or South Carolina loses the stores across the border, so it looks worse off than it is.

**11. Language.** 99.5% of reviews with eight or more words are English. Very short non-English reviews ("Bueno") go through the English model as written. We didn't translate, because the gain is too small to matter.

**12. Foot traffic is partial and licensed.** The drive-time markets can weight stores by Advan foot-traffic visits (2,806 of 4,452 stores matched, mostly chains). It is optional; every result holds without it, and most gaps get larger. The data is licensed, so store-level foot traffic stays on the lab server and is not in the Drive folder.

---

## 5. What's left

**Needs people (this week):**
- [ ] The hand labeling on Wednesday, run by Caroline on her own copy of the labeling page (only a page's owner can download its results). Her guide covers it.
- [ ] Run `run_after_labeling.sh`. It produces the final scores, refreshes the Drive folder, and takes about 1.5 hours.

**Needs outside answers:**
- [ ] Ask Teresa for her Raleigh tract list, the store set behind her "distance to nearest store", and whether her store z-scores were taken before or after pooling. Her data file and her thesis text disagree on the last one.

**Worth doing, any time:**
- [ ] Check the store list against Census county counts (Step 4).
- [ ] Add stores across the state line so border tracts are scored fairly.
- [ ] Decide whether the 32-minute market stays as a sensitivity check or becomes the rural fallback. It flips the sign of the Affordability result, because a 32-minute market in a city covers most of the metro.

**Decisions that are yours:**
1. **Which pipeline is the headline?** My recommendation: report B as the main result, with A beside it as the replication of Teresa. A is what makes B credible, and B is what makes the work new.
2. **Which market size for Raleigh?** I'd present 8 minutes for Raleigh and 16 for the state, and say why.
3. **Is checkout a sixth dimension?** The labeling's write-ins and a rerun of the dimension analysis on human labels can answer it.

---

## 6. Where everything is

**GitHub** (private): `suyashh77/nc-aspatial-food-access` holds everything in one place: code, these documents, results and, through `fetch_data.sh`, the database. It's the starting point for whoever continues. Suyash will add collaborators; it should be transferred to you or a lab account so it outlives his account.

**Google Drive** holds the results, the documents, the code and an archive copy of the database. `README` at the top of the folder says what each subfolder is. `1_Read_Me_First/FILE_MANIFEST` lists every folder on the server and whether it went to Drive.

**The lab VM** (`manabe.ie.ncsu.edu`, `/home/scsapre/aspatial_food_access_research/`) holds everything, including the large intermediate files (embeddings, roughly 5 GB) and the licensed foot-traffic data. The pipelines run here.

**My PC** holds the scraping code (the store frame, the scraper and the repair scripts). It should be zipped and uploaded to Drive, minus the `Auth/` folder, which holds live Google logins and must never be shared.

**GitHub** holds geomatch, the record-matching tool: https://github.com/suyashh77/geomatch.
