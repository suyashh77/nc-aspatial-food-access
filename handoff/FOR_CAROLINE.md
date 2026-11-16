# Hand labeling and using the data

For Caroline. Written November 15, 2026, at handoff.

This guide covers three things:

1. How to run the hand labeling.
2. What to do when it's finished, and what the results will look like.
3. How to use the scores in a study without getting caught by the data's weak spots.

Everything in the pipeline that a computer can do is done. The one thing left is the part only people can do: checking the model's topics and labeling a set of reviews by hand. Until that happens, every score is marked **provisional**.

Two other documents sit next to this one. *FOR_DR_RACHUNOK* explains the project, the methods and the data's problems in full. *FOR_CLAUDE* is the technical handoff, written so a Claude session can pick up the code.

---

## 1. Why the hand labeling matters

The pipeline sorts each Google review into the five dimensions of access. It does this in two steps. First, a model groups the 975,000 written reviews into 83 topics ("parking", "prices", "rude cashiers"). Then each topic is matched to the dimension whose definition it sounds most like.

Two things in that chain need a human:

- **Some topics are junk.** A topic made of "great store, love it" has no subject. Its reviews shouldn't count toward any dimension. Right now a rough list I drew up decides which topics are junk. People need to make that call.
- **We need an answer key.** To know whether the model labels reviews correctly, we need reviews labeled by people. Today the answer key is a stand-in that Claude labeled. Every accuracy number in the project rests on it, and none of them should be reported until people redo it.

The single most important call is **Topic 0**. It holds 58,000 reviews about "the grocery store", so its label looks generic. But many of its reviews are about selection ("they have everything"), which is Availability. The rough list marks it as junk. If people agree, Availability loses much of its best evidence.

---

## 2. Before Wednesday: publish your own copy of the labeling page

The labeling page is a web page on claude.ai. Suyash's copy (https://claude.ai/artifact/7cCD2HedpJ97FgEmnLo2uH) belongs to his account, and only the owner of a page sees its Results tab, where the labels are downloaded. So **you publish your own copy and run the labeling on it.** Don't use Suyash's link for labeling. Nobody has labeled on it, so nothing is lost.

1. Get the three page files from the Drive folder, `5_Human_Labeling/web_labeler_page/`: `labeler.html`, `data.js` and `data_b2.js`.
2. Open Claude (claude.ai or Claude Code), give it those three files and ask:

   > Publish labeler.html as an artifact, with data.js and data_b2.js as supporting files. It needs the database, user identity and downloads capabilities.

   You get your own link, and you're the owner.
3. Add every rater by email as an **Editor** from the page's Share menu. The public link must be off, or Editor access won't stick.
4. Test it: open your link from a rater's account, label one review, and check that it saves. If the page says "view-only", that person isn't an Editor yet. That test label counts as a real one, so have the rater carry on from there.
5. Write your link at the top of `FOR_CLAUDE` (Section "Hand labeling") so the next person can find it.

## 3. Running the labeling session

### What raters do

The page has a Start here tab that explains everything. Raters type their initials once (use different initials for each person, or the exports can't tell two raters apart). Then there are two tasks.

**Task 1: Label reviews.** 260 reviews, and the page wants 5 different people on every review. A rater reads one review and, for each dimension, picks one of: not discussed, positive, negative, or mixed. If the review says nothing about access at all, they tick "No access content." There is also a free-text box for any important subject that fits none of the five. About 30 seconds a review. Raters never see the stars, the store or the model's guess.

The page hands each rater the next review that still needs raters. So five people each doing all 260 is about two hours each. More people means less each.

**Task 2: Check topics.** 83 topics in about 25 groups of near-duplicates. For each group, the rater reads the keywords and example reviews, marks topics that are too vague to use, moves any topic that's in the wrong group, and scores the group 0 to 3 on each dimension. The score is about what the reviews discuss, not whether they're positive: "prices are crazy" and "great deals" both score 3 on Affordability. Allow 1.5 to 2 hours.

**Task 3 (optional): Check sentence topics.** The same job for the improved pipeline's sentence-level topics. About 1.5 hours. Skip it if time is short. The improved pipeline then keeps using its automatic filter, which caught only 4 vague topics, so its accuracy numbers stay softer.

### The rules raters should follow

These are shown on the page too. They exist so raters agree with each other.

| A review about... | goes under |
|---|---|
| Stock, empty shelves, selection, variety, "they have everything", wanting a store nearby | Availability |
| Location, distance, parking, traffic, the bus, getting in the door | Accessibility |
| Hours, EBT/SNAP/WIC, pickup and delivery, checkout lines, layout, finding items, policies | Accommodation |
| Prices, value, deals, coupons, being overcharged | Affordability |
| Freshness and quality, cleanliness, how staff treat people, safety, cultural fit | Acceptability |
| "Great store!" with nothing else | none of them |

Two of these calls differ from Teresa's original definitions. Teresa's text files selection under Acceptability, and the page files it under Availability. The page also puts checkout and pickup under Accommodation. We chose the page's rules because they follow the literature's definitions. Whatever the raters decide in practice is exactly what the project needs to learn.

### How many people you need

| Task | Minimum for the results to count | Better |
|---|---|---|
| Label reviews | At least 3 raters on at least 200 of the 260 reviews. Below that, the analysis keeps using Claude's stand-in labels. | 5 raters on all 260 |
| Check topics | 1 person (their scores become the answer) | 3 people, so a majority decides |
| Sentence topics | Optional | 1 to 3 people |

The import uses majority vote. A dimension counts for a review when more than half its raters mark it. A topic is junk when more than half the people who checked it say so.

---

## 4. After labeling: turning labels into final scores

This takes about 1.5 hours of computer time and one command. Everything is also in the private GitHub repo `suyashh77/nc-aspatial-food-access` (clone it and run `./fetch_data.sh`), but the easiest place to run it is the lab VM (`manabe.ie.ncsu.edu`), in `/home/scsapre/aspatial_food_access_research/`. If you've never used the VM, ask Dr. Rachunok or the department for access first, or hand this section to whoever runs code for the group (Claude can do it from Claude Code on the VM).

### Step 1. Download the labels

On your copy of the page, open the Results tab and click:

- *Download review labels (all raters, CSV)*. This saves `gold_labels_all_raters.csv`.
- On the Check topics page, pick "Whole-review topics", go back to Results and click *Download topic scores (all raters, CSV)*. This saves `step5_manual_scores_all_raters.csv`.
- If anyone did Task 3, switch to "Sentence topics" and download again. This saves `b2_manual_scores_all_raters.csv`.

Put those files together in one folder on the VM, for example `~/labels_2026-11-19/`.

(If a Claude session is doing this, it can skip the downloads and read the page's database directly. *FOR_CLAUDE* says how.)

### Step 2. Run the script

```bash
cd /home/scsapre/aspatial_food_access_research
./run_after_labeling.sh --csv ~/labels_2026-11-19
```

It prints one line per step with "ok" and the time taken, and stops with the error if a step fails. It does, in order:

1. Imports the labels and works out the majority vote.
2. Recomputes store scores with Teresa's method (both versions), for the state and for Raleigh.
3. Recomputes the census-tract scores and the equity regressions.
4. Reruns the decision tests on the human answer key.
5. Retunes the improved pipeline's definitions on the human answer key, then reruns it, the drive-time markets and the Raleigh comparison.
6. Rebuilds the review-level file and the Google Drive folder.

The script was tested end to end on November 15 with five fake raters, in a separate copy of the project.

### Step 3. Check what changed

Open `google_drive_upload/LABEL_STATUS.txt`. It should say `Labels: HUMAN`. The score files lose `_PROVISIONAL` from their names.

Then look at `5_Human_Labeling/imported_labels/step5_import_meta.json`. It lists who labeled what and how well raters agreed. The agreement measure is Fleiss' kappa, one number per dimension. Above 0.6 is usually called substantial. Below 0.4 means the raters read the rules differently, and the rules need another look before anyone trusts the labels.

Finally, upload the rebuilt `google_drive_upload/` folder to Drive in place of the old one.

### What the results look like

Here are the files the import writes (column names are real; the rows are made up to show the format):

`gold_consensus.csv`, one row per review that has at least 3 raters:

| gid | id | n_raters | no_access_content | Availability | ... | Affordability | Availability_votes | ... |
|---|---|---|---|---|---|---|---|---|
| 17 | 1112330 | 5 | False | + | | - | 4 | |

A `+`, `-` or `m` (mixed) means the majority said the review discusses that dimension. Blank means it doesn't. `_votes` is how many raters marked it.

`step5_manual_scores.csv`, one row per topic, in Teresa's format:

| Topic | manual_topic_status | manual_invalid_reason | manual_Availability | ... | n_raters | n_invalid_votes |
|---|---|---|---|---|---|---|
| 0 | valid | | 3 | ... | 3 | 1 |
| 44 | invalid | too_generic | 0 | ... | 3 | 3 |

Only topics marked `valid` feed the store scores. A topic marked `needs_review` (raters kept it but couldn't agree on its status) is left out too, so check the file for any.

The scores themselves keep the same columns as now (see *DATA_DICTIONARY*). Expect them to move. The biggest moves will be on Availability, Accessibility and Accommodation, the dimensions with the fewest reviews, and above all if Topic 0 is kept.

---

## 5. Using the data in a study

### Which file answers which question

| Question | File (in `google_drive_upload/`) |
|---|---|
| Which grocery stores exist in NC? | `2_Store_List/NC_Grocery_POIs_v2.xlsx` |
| How does a store score on each dimension? | `3_Scores/store_scores_B_improved*.csv` (recommended) or `store_scores_A_teresa_method*.csv` (Teresa's exact method) |
| How good are the stores a neighborhood can reach? | `3_Scores/tract_scores_B_drive_time_*.csv` (16-minute drive) or `tract_scores_A_1mile*.csv` (Teresa's 1-mile decay) |
| What did people actually write? | `4_Review_Level/NC_reviews_with_dimensions*.parquet` |
| Raleigh, compared with Teresa | `6_Raleigh_Case_Study/Raleigh_case_study*.xlsx` |

The review-level file has every review (1.94 million) with its store, its text, and the labels from every pipeline. It's a Parquet file because a CSV would be over a gigabyte. Read it in R with `arrow::read_parquet()`, in Python with `pandas.read_parquet()`. A 2,000-row CSV preview sits next to it for a first look.

### Teresa's method (A) or the improved one (B)?

Use **A** when the point is to compare with Teresa's Raleigh study. Within A, the "frozen" version runs Teresa's own Raleigh model on our reviews, and it matches her store rankings closely (rank correlation 0.90 to 0.95 on her 42 matched stores).

Use **B** for anything new. It reads reviews sentence by sentence, so a review that praises prices and complains about parking counts for both. It judges good or bad from the text, not only the stars. And it stops one-review stores from topping the rankings. Each change was tested against A. The numbers are in `PIPELINE_GUIDE.md`, section 3.

Report which one you used. For headline results, show both.

### Things that will bite you

These come up in any study with this data. *FOR_DR_RACHUNOK* has the full list.

- **Most stores have few reviews on the smaller dimensions.** The median store has 60 reviews about Acceptability, but only 5 about Accommodation and 7 about Accessibility. Each score file has an `_n` or `_n_reviews` column. Use it. For a store-level claim, require at least 20 reviews on that dimension, or use the 95% interval columns.
- **In Teresa's method, the "best" stores are often stores with one or two 5-star reviews.** That comes from how her model handles stores with tiny samples. Pipeline B fixes it. If you use A's store rankings, filter on review counts.
- **Dollar stores are a quarter of the list** (1,001 of 4,442). They are in scope on purpose, because for many rural residents they are the grocery store. But averages over "all stores" are partly averages over Dollar General. The `category` column lets you split them out.
- **Big stores hold only their newest reviews.** 257 stores have more than 1,500 reviews on Google, and for those we hold the most recent ones, not their whole history. They carry a quarter of all reviews. The flag is `store_holds_newest_only`.
- **Review dates are rough.** Google says "3 years ago", not a date. `review_year_approx` is good to about a year. Don't build a monthly trend on it.
- **A few stores don't belong.** Three have addresses outside NC (store IDs 5379, 9368 and 8852), and 37 have a restaurant as their main Google category. Drop them, or check the restaurant ones by hand.
- **Border tracts look worse than they are.** Only NC stores are in the data, so a tract near Virginia or South Carolina loses the stores across the line.
- **Rural tracts often have nothing within reach.** 162 tracts (489,418 people) have no store in the study within a 16-minute drive. Their tract scores are blank or come from a 32-minute fallback (`_fallback` columns). That is a real gap, not a low score.

### Reading a regression coefficient

The tract regressions put each tract's score on the share of residents who are Black (0 to 1) and median income (in $10,000s), weighted by population. The scores are standardized, so coefficients read in standard deviations. One wrinkle: in Teresa's 1-mile version a tract score is an average of store z-scores (so the unit is the spread across stores), while in the drive-time versions tract scores are standardized across tracts (the unit is the spread across tracts). Say which one you're reporting.

Because % Black is a share, a coefficient of −2.0 means a tract that is 10 points more Black (0.10) scores about 0.2 SD lower, holding income fixed.

---

## 6. For the Raleigh talk

Use the frozen version of Teresa's method to show the replication, and 8-minute drive-time markets for the equity results. All numbers below are provisional until the labels are in. They come from `6_Raleigh_Case_Study/`.

**The replication.** On Teresa's 42 stores that we matched, her model run on our newer reviews ranks the stores much as she did:

| | Availability | Accessibility | Accommodation | Affordability | Acceptability |
|---|---|---|---|---|---|
| Rank correlation with her scores | 0.90 | 0.95 | 0.93 | 0.91 | 0.95 |

A model refit on all of NC agrees with her only on Acceptability (0.93). On the other four it picks different reviews, so it measures them differently. That is a finding about the method, and worth one sentence in the talk.

**Her Affordability result.** Teresa found that tracts with a higher Black share had *more* affordable stores (coefficient +1.64). From her own 49 store scores we rebuild that number almost exactly (+1.65). When every Wake County store is included, it reverses in Teresa's method (−0.41) and disappears in Pipeline B (−0.05, not significant). The fair way to say it: her result depends on which 49 stores were in the study. We don't have her exact list of tracts, so this is an approximate reproduction, not proof she was wrong.

**Why 8 minutes, not 16.** A 16-minute drive from the median Wake tract reaches 105 stores, nearly the whole metro, so neighboring tracts share the same market. An 8-minute drive reaches a median of 21 stores, and the differences show up:

| Wake County, 8-minute markets | Teresa's method (A) | Improved (B) |
|---|---|---|
| Availability | −2.00* | −2.03* |
| Accessibility | −0.68 | −0.94 |
| Accommodation | −1.36* | −1.63* |
| Affordability | −1.79* | −0.59 |
| Acceptability | −1.44* | −1.30* |

\* p < 0.05. Coefficient on % Black (as a share), 211 tracts, stores weighted by foot traffic.

In plain terms: in Wake County, a tract whose population is 10 points more Black has nearby stores that score about 0.2 SD lower on Availability, and about 0.13 to 0.16 SD lower on Accommodation and Acceptability. Both methods agree on those three. Affordability depends on the method, so don't lead with it. Without foot-traffic weights the gaps are larger, and Accessibility becomes significant too.

**Stores people will recognize.** Lidl leads Affordability in every version. Harris Teeter and Publix lead Acceptability and Accommodation.

---

## 7. If something goes wrong

| Problem | What to do |
|---|---|
| A rater sees "view-only" | Add them by email as Editor in the Share menu, with the public link off |
| You can't see the Results tab | You're on someone else's copy. Use the one you published (Section 2) |
| Two raters used the same initials | The Results tab warns you. The exports still separate them by account, so nothing is lost |
| The script stops at "No data/step5_manual_scores.csv" | The topic-check CSV wasn't in the folder. Download it (Section 4, step 1) |
| The script stops at another step | The last 20 lines of the error are printed, and the full log is in `logs/after_labeling_<time>/`. Give that log to Claude |
| Agreement is very low on a dimension | Read the disagreements on the Results tab, tighten the rule for that dimension, and relabel those reviews |
