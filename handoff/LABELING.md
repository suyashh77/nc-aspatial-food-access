# Labeling

How to run the hand labeling and turn it into final scores.

## Why it matters

- The pipeline groups 975K reviews into 83 topics, then matches each topic to a dimension.
- Some topics are junk ("great store, love it"). A hand-made draft list decides which ones today. People need to make that call.
- Every accuracy number rests on a 260-review answer key that Claude labeled as a stand-in. People need to redo it.
- Biggest single call: **Topic 0**. 58K reviews about "the grocery store". Looks generic, but many reviews are about selection (Availability). The draft marks it junk. If raters agree, Availability loses much of its evidence.

## The plan

- Session: Wednesday, Nov 18, 2026. Suyash runs it on his page and pulls the results.
- Then Suyash runs `run_after_labeling.sh`, refreshes the docs, and hands the final folder to Caroline.

## The page right now (checked 2026-11-16)

- Live at https://claude.ai/artifact/7cCD2HedpJ97FgEmnLo2uH, owned by Suyash's account.
- Matches the files in `labeling/web_labeler_page/`. Database rules are set: each rater writes only their own labels, only the owner sees everyone's.
- All label collections are empty. Nobody has labeled.
- Sharing is set to "anyone with the link". **Turn the public link off** before adding raters, or Editor access won't stick.
- Only the owner sees the Results tab, where labels are downloaded. Don't label on two copies: labels don't move between them.

## Before Wednesday (Suyash)

1. Share menu: public link off.
2. Add every rater by email as **Editor**.
3. Test from one rater's account: label one review, check it saves. "View-only" means they aren't an Editor yet. The test label counts.

## If someone else ever reruns the labeling

Only needed if the labeling is redone after Suyash's account is gone. They publish their own copy and own its results.

1. Take the three files from `labeling/web_labeler_page/`: `labeler.html`, `data.js`, `data_b2.js`.
2. In Claude (claude.ai or Claude Code), give it the three files and ask:
   > Publish labeler.html as an artifact, with data.js and data_b2.js as supporting files. It needs the db, user and downloads capabilities. Use these db rules: each of gold, topics, topics_b2 and claims is readable by admins and writable by the owner, and each `<collection>/{self}` doc is readable and writable at interact level (claims readable at interact for everyone).
3. Share menu: public link off, then add every rater by email as **Editor**.
4. Test from a rater's account: label one review, check it saves. "View-only" means they aren't an Editor yet. The test label counts, so they carry on.
5. Write the new link in this doc and in `handoff/FOR_CLAUDE.md` in the repo.

## What raters do

- Type initials once, top right. Different initials per person.
- **Task 1: label reviews.** 260 reviews, 5 raters each. For each dimension: not discussed / positive / negative / mixed. Or tick "No access content". ~30 sec per review. Raters never see stars, store, or the model's guess. Page hands out the next review that still needs raters. 5 people ≈ 2 h each.
- **Task 2: check topics.** 83 topics in ~25 groups. Mark vague topics invalid, move misplaced ones, score each group 0–3 per dimension on what reviews are *about*, not whether they're positive. 1.5–2 h.
- **Task 3 (optional): sentence topics.** Same job for Pipeline B's topics. ~1.5 h. Skip if short on time.

Rules raters follow (also on the page):

| Review about... | Goes under |
|---|---|
| Stock, empty shelves, selection, variety, "has everything", wanting a store nearby | Availability |
| Location, distance, parking, traffic, bus, getting in | Accessibility |
| Hours, EBT/SNAP/WIC, pickup, delivery, checkout lines, layout, policies | Accommodation |
| Prices, value, deals, coupons, overcharged | Affordability |
| Freshness, quality, cleanliness, staff treatment, safety, cultural fit | Acceptability |
| "Great store!" and nothing else | none |

- Two calls differ from Teresa's text: selection goes to Availability (hers: Acceptability), checkout and pickup go to Accommodation. These follow the literature.

## How many raters

| Task | Minimum to count | Better |
|---|---|---|
| Label reviews | 3 raters on 200+ of 260. Below that the code keeps Claude's stand-in | 5 on all 260 |
| Check topics | 1 person | 3, majority decides |
| Sentence topics | optional | 1–3 |

- Majority vote. A dimension counts for a review when more than half its raters mark it. A topic is junk when more than half say so.

## After labeling

1. Download from the Results tab (owner only):
   - *Download review labels (all raters, CSV)* → `gold_labels_all_raters.csv`
   - On Check topics pick "Whole-review topics", back to Results, *Download topic scores (all raters, CSV)* → `step5_manual_scores_all_raters.csv`
   - If anyone did Task 3, switch to "Sentence topics" and download again → `b2_manual_scores_all_raters.csv`
2. Put them in one folder on the lab VM, e.g. `~/labels_2026-11-19/`.
3. Run:
   ```bash
   cd /home/scsapre/aspatial_food_access_research
   ./run_after_labeling.sh --csv ~/labels_2026-11-19
   ```
   ~1.5 h. Prints one "ok" line per step, stops on the first error. Logs in `logs/after_labeling_<time>/`.
   - Alternative: a Claude session signed in as the page owner can pull the database directly with `ArtifactData list` on `gold`, `topics`, `topics_b2` into `NC Pipeline/data/web_labels`, then `./run_after_labeling.sh --json "NC Pipeline/data/web_labels"`. Same output.
4. Check:
   - `google_drive_upload/LABEL_STATUS.txt` says `Labels: HUMAN`. Score files lose `_PROVISIONAL`.
   - `step5_import_meta.json`: Fleiss' kappa per dimension. Above 0.6 is good. Below 0.4 means raters read the rules differently. Tighten the rule and relabel before trusting it.
   - `step5_manual_scores.csv`: any `needs_review` topics are left out of scores. Look at them.
5. Update `PIPELINE_GUIDE.md` numbers and these docs, rebuild the deck if numbers changed, commit, rerun `build_drive_package.py --zips`, upload to Drive.

What to expect: scores will move most on Availability, Accessibility and Accommodation (fewest reviews), and most of all if Topic 0 is kept.

## If something goes wrong

| Problem | Fix |
|---|---|
| Rater sees "view-only" | Add as Editor by email, public link off |
| No Results tab | You're on someone else's copy |
| Two raters used the same initials | Results tab warns you. Exports still separate them by account |
| Script stops at "No data/step5_manual_scores.csv" | Topic CSV missing from the folder. Download it |
| Script stops elsewhere | Give the log in `logs/after_labeling_<time>/` to Claude |
| Low agreement on a dimension | Read disagreements on Results, tighten the rule, relabel |
