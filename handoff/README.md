# NC aspatial food access: data and results

Every grocery store in North Carolina (4,442), the Google reviews written about them (1.94 million), and scores for each store and census tract on the five dimensions of food access: Availability, Accessibility, Accommodation, Affordability and Acceptability. The method scales Teresa Groton's Raleigh thesis to the whole state. NC State ISE, Dr. Benjamin Rachunok's group. Handed off November 15, 2026.

**Status:** check `LABEL_STATUS.txt`. If it says PROVISIONAL, the scores are waiting on a hand-labeling step, and files carry `_PROVISIONAL` in their names. Once it's done and the build is rerun, the status reads HUMAN.

## Start with one of these

| You are | Read |
|---|---|
| Dr. Rachunok, or anyone who wants the whole story and the data's weak spots | `1_Read_Me_First/FOR_DR_RACHUNOK` |
| Running the hand labeling, or using the scores in a study | `1_Read_Me_First/FOR_CAROLINE` |
| A Claude session or a programmer picking up the code | `1_Read_Me_First/FOR_CLAUDE` |

The slides covering the whole project are `1_Read_Me_First/Final_Presentation.html` (open it in a browser and use the arrow keys) and the same deck as a PDF.

## What's in each folder

| Folder | Contents |
|---|---|
| `1_Read_Me_First` | The three guides, the data dictionary, the file manifest, the presentation |
| `2_Store_List` | The final list of grocery stores |
| `3_Scores` | Store and census-tract scores from both pipelines, plus the equity regressions |
| `4_Review_Level` | Every review with its store and labels (Parquet), plus a CSV preview |
| `5_Human_Labeling` | The labeling page, the reviews and topics to label, and imported labels |
| `6_Raleigh_Case_Study` | Raleigh compared with Teresa's thesis |
| `7_Methods_and_Reference` | The full pipeline guide, the scraping record, decision-test results, Teresa's thesis |
| `8_Code` | The whole GitHub repo as one zip: code, docs, results, Teresa's files. Unzip it, then copy the large files from `9_Database` and `4_Review_Level` in (see the zip's README) |
| `9_Database` | The review database (SQLite) and the side databases of stores ruled out |

`PACKAGE_CONTENTS.csv` lists every file with its size and checksum. Column meanings are in `1_Read_Me_First/DATA_DICTIONARY`.

Store-level foot-traffic data (Advan, SafeGraph) is licensed and stays on the lab server, not here.
