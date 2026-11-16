# NC Aspatial Food Access: handover

Every grocery store in North Carolina (4,442), their Google reviews (1.94M), and scores for each store and census tract on the five A's of food access: Availability, Accessibility, Accommodation, Affordability, Acceptability. Scales Teresa Groton's Raleigh thesis to the whole state. NC State ISE, Dr. Benjamin Rachunok's group. Handed off by Suyash Sapre, Nov 2026.

## Status

- Data collection: done and frozen (freeze v2, 2026-10-24).
- Both pipelines (A = Teresa's method, B = improved) run end to end.
- All scores are **provisional** until the hand labeling is done. See `LABEL_STATUS.txt`.
- Labeling: Wed Nov 18, run by Suyash on his page (live, empty so far). Then final scores, then the project goes to Caroline. See `docs/LABELING.docx`.

## Read in this order

1. `docs/HANDOVER.docx`: the project, the pipeline, what's left, where things live.
2. `docs/LABELING.docx`: how to run the labeling and turn labels into final scores.
3. `docs/USING_THE_DATA.docx`: which file answers which question, and the traps.
4. `docs/DATA_DICTIONARY.docx`: every column.
5. `docs/PIPELINE_GUIDE.docx`: the full method and every test result. Reference only.

## Folders

| Folder | What's in it |
|---|---|
| `docs/` | The Word docs above, the scraping record (`scraping/`), Teresa's thesis. Plain-text copies of the docs for Claude or GitHub are in `markdown/` |
| `presentation/` | Final slides. Open the HTML in a browser, arrow keys to move, N for notes. PDF is the same deck |
| `labeling/` | The labeling web page (`web_labeler_page/`), what raters label (`inputs/`), and where imported labels go (`imported_labels/`) |
| `data/` | Store list, scores, review-level file, Raleigh results, databases, `pipeline_working_files.zip`. Inner layout is what `place_data.sh` expects, so don't rearrange it |
| `results/` | Decision-test and Pipeline B test outputs (JSON), Raleigh workbook |
| `code/` | Backup zip of the code repo. The live code is on GitHub: `suyashh77/nc-aspatial-food-access` (private) |
| `_archive/` | The earlier long-form guides, Word copies, old file list, duplicate files. Nothing here is needed |

Licensed foot-traffic data (Advan, SafeGraph) is not here. It stays on the lab VM.
