# Handoff: NC grocery stores and their Google reviews

Data collection closed on 2026-10-24. This page gives you the state of things in ten minutes.
For every decision and its reason, read [STATUS.md](STATUS.md) next.

## What this is

A dataset of every grocery store in North Carolina, plus the Google reviews written about each one.
It feeds a dissertation on aspatial food access at NC State ISE. That work measures access through
what people experience at a store (price, how they're treated, what's on the shelves), using
Penchansky & Thomas's five dimensions, instead of distance alone.

## The deliverable

| | |
|---|---|
| File | `Data/Step_5_master_reviews_FROZEN_2026-10-24.db` (SQLite, read-only) |
| Checksum | `Remediation/master_freeze_manifest_v2.json` (SHA-256) |
| Stores | 4,442 (4,400 complete, 42 short) |
| Reviews | 1,941,538, of which 977,398 have text, across 590 cities |
| Store table | `Data/exports/NC_Grocery_POIs.xlsx`. **It was built from freeze v1 (4,447 stores) and needs rebuilding from v2.** |

Tables: `stores`, `reviews`, `quarantined_reviews` (29,321 set-aside reviews, never deleted), `accepted_flags`.
Side files: `Step_5_out_of_scope.db` (gas stations and other out-of-scope stores), `Step_5_closed_stores.db`, `Step_5_excluded_stores.db`.

## How it was built, in six steps

1. **Store universe** (`Frame/`). I combined USDA SNAP retailers, Overture Maps and OpenStreetMap, then checked
   them against Data Axle, Dewey (SafeGraph) Places and Census business counts. geomatch (a sibling checkout,
   see `requirements.txt`) matched records by distance and name similarity. SNAP is trusted for the address,
   Overture for the name, Google for the place ID. 91,461 records became a 36,820-store frame.
2. **Google probe** (`Frame/probe.py`). I searched each candidate store's name and address on Google Maps,
   recorded its place ID, and scraped its category and subcategory. That checked 16,751 stores.
3. **Scope.** I hand-ruled the categories (`Frame/category_ruleset.csv`, about 300 rulings) and ran manual rounds.
   Chains with more than five locations are in automatically. Out: gas stations, convenience stores, closed stores,
   farmers' markets, produce stands, small general-purpose shops with some groceries, and restaurants that sell groceries.
4. **Review scraping** (`Scraper_v2/linux_scraper`). Python and Playwright drive Chromium on NC State Linux machines,
   one scraper per machine. Each machine runs a batch database built by `Remediation/r09` and shipped
   as a kit by `Deploy/build_kit.py`.
5. **Merge and repair** (`Scraper_v2/linux_scraper/merge_results.py`, `Remediation/r00`–`r18`). Results merge into
   the master with a duplicate check. Then `r10` sets targets from Google's live counts, `r17` restores confirmed
   reviews, `r15` applies the completion rule, and `r00` validates.
6. **Freeze.** A `VACUUM INTO` copy plus a checksum manifest.

## Rules that still bind

- **Completion rule (final):** a store's target counts as at most 1,500. A store is complete when it is missing
  no more than 10 reviews (target 100 or under), 20 (over 100), 50 (over 500) or 100 (over 1,500).
  The rule lives in `Remediation/r13_rulings.py` (`TOLERANCE_TIERS`) and `r15` applies it. Stored targets stay Google's true count.
- Never delete reviews. Move them to `quarantined_reviews` or a side database and record a reason.
- The frozen file is read-only. Changing the data means a new freeze, a new manifest and an updated STATUS.md.
- One copy of every piece of code. Kits are built outputs; fix `Scraper_v2/`, add a test, rebuild.
- Dry-run every merge and every writer first. Every writer backs up to `Data/backups/` on its own.
- `python -m pytest -q` stays green (329 tests).

## Known issues, worst first

1. **Big stores are truncated.** About 250 stores have more than 1,500 reviews on Google. For those the dataset
   holds the newest reviews, not their full history. Flag them in any per-store analysis.
2. **The store table is out of date.** `NC_Grocery_POIs.xlsx` reflects freeze v1.
3. **Rural stores are under-covered.** Small rural stores are the ones most often missing from commercial maps.
4. **42 stores are short.** 40 are 21–25 reviews short after repeated passes, and 2 (Dollar General 2444,
   Save A Lot 3860) wait on a manual check. That's 955 reviews in all.
5. **109 stores have a target of exactly 1,500**, an old capped number, so their true size is unknown.
6. **Review dates are relative** ("2 years ago"). Exact dates have to be estimated from `scraped_at`.
   Reviewer name is not an ID ("Chris" reviews 240 stores).
7. **The validator exits HIGH** for points 4 and 5. Its one "double scrape" flag (Wegmans Morrisville, 8259)
   is a stale target with no duplicates.
8. **Code fork.** The manabe machine ran a hand-patched scraper. `Scraper_v2/` stays canonical. The box's copy and a
   diff are in `Archive/manabe_box_code_20261024/`. Read the diff before you ever scrape again.
9. **Don't run `r03_clean_deterministic.py --apply`.** Its status step predates the completion rule and would
   undo it on about 1,300 stores.

## What to do next

1. **Analysis export.** One row per review, joined to its store, in the shape of
   `Data/reference/Teresa_Sample_Output.csv`, with flags for capped, short and Google-limited stores. STATUS.md §6 has the spec.
2. Rebuild `NC_Grocery_POIs.xlsx` from freeze v2.
3. Check the final store list against the Census counts (`Frame/reports/census_validation.csv`).
4. Settle the two Round 7 manual stores.

## Where things are

| Path | What |
|---|---|
| `STATUS.md` | the full record: decisions, numbers, history |
| `Frame/` | store universe and Google probe |
| `Scraper_v2/` | the review scraper (canonical) |
| `Remediation/` | audit, repair and ruling scripts; freeze manifests |
| `Deploy/` | kit builder |
| `Data/` | all data (gitignored); `Data/README.md` explains each file |
| `Auth/` | Google login files. Never commit or share them. They only work on the PC they were made on. |
| `../Gmaps Scraper - external data/` | `Dr.ben_data/` and `dewy data/`, moved out of the repo; override the path with `GMAPS_EXTERNAL_DATA` |
| `Archive/DELETED_2026-10-24.md` | what the handoff cleanup deleted, if something looks missing |

## Commands

```bash
python -m pytest -q                                   # all tests
python Remediation/r00_validate.py --fail-on HIGH     # validate the master
python Remediation/r15_tolerance.py                   # completion rule, dry run (--apply to write)
cd Scraper_v2 && python -m linux_scraper.merge_results ../Data/returns/<dir>/<file>.db \
    --master ../Data/Step_5_master_reviews.db --dry-run
```
