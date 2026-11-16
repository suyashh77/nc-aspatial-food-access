# Aspatial Food Access: From Raleigh to All of North Carolina

**A from-zero guide to Teresa Groton's method, the NC data, and the plan to scale it**

*Audience: someone new to this project, to food-access research, and to transformers/LLMs.
Read top to bottom: each part builds on the last. Videos are in Part 9.*

---

## Part 1. The big picture in plain English

**The problem.** We want to know which neighbourhoods have good access to food. The usual answer is
the "food desert" map: draw a circle around each grocery store, and if your home is more than ~1 mile
from any store (urban) or 10 miles (rural), you're in a desert.

**Why that's not enough.** Distance is only one part of access. A store 0.5 miles away is no help if
it is:
- too expensive;
- dirty, or has rude staff;
- closed when you get off work;
- doesn't take SNAP/WIC;
- has empty shelves.

Researchers call these the **aspatial** (non-distance) parts of access. They are hard to measure: you
need surveys or in-store audits, which are slow and expensive and don't scale.

**Teresa's idea.** People already write about all of this, for free, in **Google Maps reviews**.
"Prices are crazy here", "open till midnight, perfect after my shift", "the cashier was so rude":
together they form an *accidental food-access survey* with millions of responses, each tied to a
real store at a real address.

**Her method in one sentence.** Use AI language models to sort tens of thousands of reviews into
the **five dimensions of access**, score every store on each dimension, put those scores on a map, and
ask whether poorer or Black neighbourhoods get worse stores.

**What we are doing.** Teresa did this for **49 stores in Raleigh** (25,125 reviews). Over the past
year our Google Maps Scraper collected **933,865 text reviews for 4,495 food stores across all 100
NC counties**. The job now is to run her method on the whole state, and improve it where we can.

---

## Part 2. The framework: the 5 A's of access

The framework comes from Penchansky & Thomas (1981). It was originally about health-care access and
was later adapted to food (see `Teresa Files/Konapur.pdf`, *"The 5 A's Approach for Contextual
Assessment of Food Environment"*).

| Dimension | Plain meaning | Example review |
|---|---|---|
| **Availability** | Is there enough food, and is it stocked? | "Lots of empty shelves. Did not find what I need." |
| **Accessibility** | Can I physically get there? Distance, parking, bus | "Easy to get to, easy to get in and out of." |
| **Accommodation** | Does the store fit my life? Hours, payment, SNAP/WIC, dietary needs | "Love that it's open to 12am, I work nights." |
| **Affordability** | Prices and value for money | "Great deals." / "Way overpriced." |
| **Acceptability** | Does it meet my standards? Freshness, cleanliness, staff | "Disgusting, expired milk on the shelf." |

**Background reading in `Teresa Files/`:**
- **`When are food deserts.pdf`** (Widener 2014): access changes with time of day.
- **Widener 2018**: "Retiring the food desert metaphor". It is in the bib, not in the folder. Its
  argument is that a pure-distance view "overemphasizes space in a multidimensional problem".
- **`Caspi review.pdf`** (2012): a systematic review of food environment and diet. It shows
  the traditional measures (GIS, audits, surveys) and their weak links to health.
- **`Turner review paper.pdf`** (2021): which access dimensions relate to fruit and vegetable intake.
- **`Sawyer dynamic system review.pdf`** (2021): food access as a complex system with five
  subsystems.
- **`Konapur.pdf`** (2022): the 5 A's applied to food.

---

## Part 3. The AI concepts you need, from zero

### 3.1 Embeddings: turning text into coordinates
A computer can't compare meanings directly, so we turn each piece of text into a list of numbers (a
**vector**). An **embedding model** is trained so that texts with similar *meaning* get vectors that
point in similar *directions*. "The prices are outrageous" and "everything here is overpriced" end up
close together, even though they share no words.

Teresa's vectors have **384 numbers** ("384 dimensions"). Picture a point in 384-dimensional space.
You can't visualise it, but the maths is the same as for 2-D points on a map.

### 3.2 Transformers, BERT, and sentence-transformers
- A **transformer** is the neural-network design behind modern language AI, including ChatGPT and
  Claude. Its key trick is **attention**: when reading a word, the model looks at every other word in
  the sentence to work out its meaning in context. "Bank" means something different next to "river"
  than next to "money".
- **BERT** (Google, 2018) is a transformer trained to *understand* text rather than generate it.
- **Sentence-BERT / sentence-transformers** (Reimers & Gurevych 2019) fine-tunes BERT-style models
  so that a *whole sentence or paragraph* becomes one vector, with similar meanings close together.
- Teresa used **`all-MiniLM-L12-v2`**:
  - a small (33M-parameter), fast, English-only sentence-transformer from 2021;
  - "MiniLM" means a compressed BERT; "L12" means 12 layers.
- **LLMs vs embedding models.** An LLM (ChatGPT, Claude) *generates* text. An embedding model only
  *encodes* text into a vector. Teresa's pipeline is almost entirely embedding-based. An LLM (OpenAI
  GPT-3.5) was used only to give each topic a short human-readable name like "Parking Lot
  Satisfaction".

### 3.3 Cosine similarity: how close are two meanings?
This is the angle between two vectors:
- 1.0 means the same direction (same meaning);
- 0 means unrelated;
- −1 means opposite.

If vectors are first scaled to length 1 ("normalised"), cosine similarity is just a dot product: fast
and simple.

### 3.4 Topic modelling with BERTopic
We don't want to hand-label 25k reviews. **BERTopic** (Grootendorst 2022) groups similar reviews
automatically:
1. **Embed** every review (3.1).
2. **UMAP** squashes the 384-dim vectors down to 5 dims, keeping neighbours as neighbours, because
   clustering works badly in high dimensions.
3. **HDBSCAN** finds dense clusters, meaning groups of reviews saying similar things. Reviews that
   fit nowhere become **outliers** (topic −1). This is a feature: vague reviews like "good" get
   filtered out.
4. **c-TF-IDF** finds the words that are distinctive for each cluster, e.g. "parking, lot, spaces".
   These are the topic's keywords.
5. Optionally, an LLM names each topic from its keywords.

Teresa got **59 topics**, e.g. "Meat Department Quality", "Parking Lot Satisfaction" and "Operating
Hours Perception".

### 3.5 Zero-shot matching: topics → dimensions without training data
**Zero-shot** means classifying without any labelled examples. Teresa:
1. Wrote a **one-paragraph definition** of each dimension, phrased like a review.
2. Embedded the 5 definitions into the *same* 384-dim space as the reviews.
3. Computed each topic's **centroid**, the average vector of its reviews.
4. Measured the cosine similarity of every centroid to every definition, giving a 59×5 table.
5. Converted each row into probabilities with a **softmax**. The **temperature τ = 0.11** controls
   how sharp it is: a low τ makes the winner take most of the probability.
6. Applied the **multi-label rule**: the top dimension always counts, and a 2nd or 3rd dimension is
   added if it is within **δ = 0.17** of the top probability.
7. **Every review inherits its topic's dimension(s).**

### 3.6 From reviews to store scores
- **Signal from stars:** 1★ = −1 (bad), 2–4★ = 0 (neutral), 5★ = +1 (good). The *text* decides which
  dimension a review is about; the *stars* decide whether it's good or bad.
- **Store × dimension average.** Example: Store 12's Affordability = mean signal of its
  Affordability-labelled reviews.
- **Problem:** a store with 3 Affordability reviews gets a wild, unreliable average.
- **Fix: Bayesian partial pooling** (a hierarchical model). Each store's true score is assumed to come
  from a shared state-wide distribution. Stores with *few* reviews get pulled ("shrunk") toward the
  overall average. Stores with *many* reviews keep their own value.
  - Think of a baseball batting average after 3 at-bats versus 300.
  - Tool used: PyMC.
- **z-score:** how many standard deviations a store is above or below the average store. Example:
  Whole Foods Affordability = −2.03 (much worse than average); ALDI = +2.62.

### 3.7 From stores to neighbourhoods (census tracts)
- **Census tracts** are small areas of ~4,000 people with income and race data from the American
  Community Survey.
- **Tract score:** a **distance-decay weighted average** of nearby store scores, where nearer stores
  count more. The decay parameter is **h = 1** (slide 38). The exact functional form must be read
  from the thesis PDF.
- **WLS regression** (weighted least squares, weighted by tract population):
  `Score ~ %Black + MedianIncome + %Black×Income`.

### 3.8 Glossary
| Term | Meaning |
|---|---|
| Embedding | text → vector of numbers capturing meaning |
| Normalise | scale a vector to length 1 |
| Centroid | average vector of a group |
| UMAP | dimensionality reduction that keeps neighbours close |
| HDBSCAN | density-based clustering that allows "noise" points |
| Outlier topic (−1) | reviews that fit no topic |
| Softmax / temperature | turns scores into probabilities / controls sharpness |
| Partial pooling / shrinkage | borrow strength from the group for small samples |
| r-hat, ESS | MCMC health checks (want r-hat ≈ 1.00, ESS in the hundreds+) |
| Coherence / diversity | topic-quality metrics (hers: 0.528 / 0.709) |

---

## Part 4. Teresa's Raleigh study, step by step (what she actually did)

**Data.**
- 49 major-chain grocery stores (Food Lion, Harris Teeter, Wegmans, ALDI, Lidl, Whole Foods,
  Publix, Lowes Foods, etc.) in Raleigh city limits plus a 1-mile ring.
- Walmart, Target, convenience and corner stores were *excluded*.
- 25,125 text reviews, scraped with Apify in June 2025. A median store has 430 reviews.

| Step | Notebook (`Code Pipeline/`) | What happens | Key numbers |
|---|---|---|---|
| 1 | `1_Locating_POIs` | Find supermarkets in OpenStreetMap inside Raleigh + 1-mile ring | 128 candidates; nothing saved. The scrape was done separately in Apify |
| 2 | `2_Topic_Model_Bertopic` | Clean text (lowercase, strip punctuation, remove brand names), embed with MiniLM, run BERTopic | UMAP(30 neighbours, 5-d), HDBSCAN(min cluster 120). 59 topics, 52% outliers at fit |
| 3 | `3_Matching_Inference` | Embed definitions, build centroids, cosine → softmax(τ=0.11) → labels (δ=0.17) | 59×5 topic→dimension table |
| 4 | `4_Manual_Validation` + `apps/manual_validation_app.py` | Human scores each topic 0–3 per dimension in a Streamlit app, or marks it invalid | Her real run: **18 of 59 topics invalid** (too generic, etc.) |
| 5 | `5_Manual_Validation_Analysis` | Compare model labels to human labels | Small n. Model top-1 often called Acceptability topics "Affordability" |
| 6 | `6_Store_Aggregation_Partial_Pooling` | Star signal → store means → PyMC pooling → z-scores | 5 models, one per dimension; all converged |
| 7 | *(slides only, no code)* | Tract distance-decay scores + WLS on income and race | see Results |

**Results** (thesis defence, `Groton_Thesis.pptx`):
- **Same rating, different profile.** Two Harris Teeters, both 4.4★, have very different dimension
  profiles: one is praised for checkout lines and 24-hour opening, the other criticised for closing
  early. Star ratings hide this.
- **"Aspatial food deserts."** Add a preference ("I need at least average Affordability *and*
  Acceptability") and the nearest acceptable store goes from 0.63 to 1.93 miles: 20 → 63 minutes
  walking. Only 2 stores were at or above average on *all* five dimensions.
- **Tract regressions.** Stores perceived as more affordable sit near tracts with a *higher* % Black
  population and also near higher-income tracts. Acceptability rises with income, with a significant
  interaction.
- **Stated limitations:**
  - reviewers are not a representative sample;
  - there is a risk of fake reviews (chains mitigate this);
  - the method doesn't measure diet or health.

**Folder map.**
- `Teresa Project/Raleigh_Data/` holds the **real thesis data** (her README says so):
  - `raleigh_ring_reviews_with_dims.csv` (25,125 labelled reviews);
  - `store_scores.csv` (49 stores × 5 dimensions, pooled z-scores and tract IDs);
  - `final_embedding_model/` (the saved BERTopic model);
  - `definition_embeddings/`.
- `Teresa Project/Code Pipeline/` is a **cleaned-up demo** of the pipeline on a different, larger
  example dataset (85 stores, including Rocky Mount and Wilson). Use it for *code*, not *numbers*.
- `Teresa Project/Teresa Files/` holds the thesis PDF, the defence deck, the SRA 2025 talk and
  poster, the paper-figure deck, 5 key papers, and the full bibliography.

**Bugs and quirks found in her code** (to fix, not copy):
1. Notebook 6 labels stores with the *reviewer's* name (`name`) instead of the store name (`title`).
2. A 1e-6 SE floor means stores with 1 review are treated as perfectly measured, so they are never
   shrunk. This probably produces the extreme z-scores.
3. The topic model was trained on an older 32,751-row snapshot, while inference used 35,779 rows.
4. Several save lines are commented out, so re-running doesn't regenerate files.
5. **The definitions are inconsistent.** The thesis slides (slide 71) show *first-person* definitions
   ("There is a grocery store in my neighborhood…"). The saved `definition_text.json` and `.npy` are a
   *different, third-person* version ("It makes a difference that theres even a grocery store…").
   We need to confirm with Teresa which one produced the thesis results.

---

## Part 5. What we have for North Carolina

The data is in `Gmaps Scraper/Data/Step_5_master_reviews.db` (SQLite; `stores` + `reviews` tables).

| | Raleigh (Teresa) | NC master |
|---|---|---|
| Stores | 49 major chains | **4,495**: grocery 2,109, dollar 1,006, supermarket 216, department 178, plus ethnic, specialty, warehouse and butcher stores |
| Text reviews | 25,125 | **933,865** |
| Geography | 1 city | 597 cities, all 100 counties (county via `Frame/build/probe.db`) |
| Review date | exact (`publishedAtDate`) | only "3 years ago" + scrape date → approximate |
| Language | translated to English by Google | raw original text, no language flag |
| Reviewer ID | yes | no (name only) |

- **Linking back to Raleigh:** 42 of Teresa's 49 stores match the NC DB via `place_id` +
  `feature_id`. This lets us **re-run Raleigh from the NC data and check that we reproduce her
  scores**.
- **Scraper next step:** the scraper's own `STATUS.md` (§6 step 9) already specifies an "analysis
  export" in Teresa's column format. That is our Phase 1.
- **Bonus dataset:** `Gmaps Scraper/Dr.ben_data/dr_ben_patterns.db` holds SafeGraph/Advan weekly
  foot traffic for 7,003 NC food stores, 2017–2026. It matches Teresa's own future-work item
  ("incorporate cell phone data to see actual grocery store use").

---

## Part 6. Decisions made so far

1. **Score all 4,495 in-scope stores.** Store category is carried everywhere as a stratum.
2. **Run two topic-model arms and compare them:**
   - frozen Raleigh model (faithful zero-shot replication);
   - a new NC-trained model.
3. **Keep non-English reviews.** Translate them for the frozen (English-only) arm; embed them
   natively with a multilingual model in the new arm. Keep the original text.
4. **Consider better embedding models.** See Part 7.4: a bake-off on a hand-labelled gold set.

---

## Part 7. The plan: phases (rough outline, with a review pause after each)

Code goes in a new folder `Aspatial Access Project/NC Pipeline/`:
- plain `.py` scripts, not notebooks;
- one `config.py` for paths and parameters;
- Teresa's functions ported verbatim (`clean_text`, `sims_to_probs`, `build_frozen_centroids`, the
  star signal, the PyMC model) plus the bug fixes;
- one small `test_*.py` per phase.

The NC master DB is only ever opened **read-only**.

### Phase 0: Set up and fill the gaps (½–1 day)
- Python env:
  - sentence-transformers, bertopic, umap-learn, hdbscan;
  - pymc, arviz, nutpie;
  - geopandas, statsmodels, pandas, pyarrow;
  - CUDA PyTorch for the RTX 3050 Ti (4 GB);
  - pypdf, so we can read the thesis PDF.
- From the thesis PDF:
  - the exact distance-decay formula with h = 1;
  - the regression specification;
  - which definition text was used;
  - the sentiment-validation slides (VADER/TextBlob).
- Ask Teresa / Dr. Rachunok the open questions in Part 8.

### Phase 1: NC analysis export (1–2 days)
- `Gmaps Scraper/Scraper_v2/export_analysis.py`, following STATUS.md step 9.
  - One row per review, joined to its store, with Teresa's column names.
  - Extra columns: `county`, `category`, `approx_published_date` (scrape date minus the "N years
    ago" string), `censored_cap` (store hit Google's 1,500-review cap).
  - Output: Parquet in `Data/exports/`.
- **Language detection** (fastText `lid.176`). Plain-language note: fastText is a tiny classifier
  that guesses the language of a text.
- **Translation to English** for the frozen arm, using a local open model so we don't depend on a
  paid API. Candidates: NLLB-200-distilled-600M or MarianMT. Keep `text_original`, `text_en` and
  `lang`.
- Check: row counts reconcile with the master, no duplicate review IDs, every review has a store and
  coordinates.

### Phase 2: Reproduce Raleigh from the NC data (the trust gate, 1–2 days)
- Rebuild her Raleigh + 1-mile ring (NCDOT boundary, EPSG:2264, 5,280 ft buffer). Select those
  stores from the export and run her exact frozen pipeline.
- Compare with `Raleigh_Data/store_scores.csv` on the 42 matching stores.
  - **Pass:** Spearman rank correlation of pooled z-scores ≥ 0.8 for each dimension.
  - Some difference is expected: the scrape is newer and the review sets differ.
- Why: if we can't reproduce Raleigh, scaling to NC would only scale a bug.

### Phase 3: Gold-standard labels + embedding-model bake-off (3–5 days, partly human)
- **Gold set:** ~600 NC reviews, stratified by store category, urban/rural and language.
  - Each review is hand-labelled for which dimension(s) it discusses and whether each is positive,
    neutral or negative.
  - Two raters on ~150 overlap reviews, to measure agreement (Cohen's kappa).
  - Why: Teresa's validation rested on only 9 topics. This gives us a real yardstick.
- **Embedding bake-off on the gold set.** All candidates fit the 4 GB GPU:

  | Model | Size | Languages | Why consider it |
  |---|---|---|---|
  | `all-MiniLM-L12-v2` (baseline) | 33M | English | Teresa's model; required for the frozen arm |
  | `Qwen3-Embedding-0.6B` | 0.6B | 100+ | Strong open multilingual family (Apache-2.0); the 8B leads open MTEB multilingual |
  | `google/embeddinggemma-300m` | 300M | 100+ | Small, recent, multilingual; built for on-device use |
  | `BAAI/bge-m3` | 568M | 100+ | Widely used open multilingual workhorse (MIT) |
  | `intfloat/multilingual-e5-large-instruct` | 560M | 100+ | Solid instruction-aware baseline |

  - **Metric:** zero-shot accuracy and macro-F1 of the definition-matching rule against the gold labels.
  - **Retuning:** τ and δ are retuned for each model, because each model's cosine scale differs.
  - **Pick:** the winner becomes the embedding model of the NC arm.
  - **Caveat:** public leaderboards rank *retrieval*, not "does this review talk about prices?", so
    our own gold set decides.
- **Definitions:** test first-person vs third-person, and single-paragraph vs several short example
  sentences per dimension averaged together.

### Phase 4: Statewide topic + dimension inference, two arms (3–5 days + human validation)
- **4a, frozen arm:** MiniLM on `text_en`. Then Teresa's saved BERTopic `transform`, her topic→dimension
  table, and her 18-topic invalid list.
- **4b, NC arm:** the winning model from Phase 3 on the original text.
  - Fit BERTopic on a stratified ~150k sample (clustering all 0.9M is impractical), then assign all
    reviews to topics.
  - Expanded brand stopwords: Ingles, Dollar General, Family Dollar, Dollar Tree, Walmart, Target,
    Costco, Compare Foods, Save A Lot, and so on.
  - Topic names come from a local LLM or Claude.
  - **Human validation of the new topics** in Teresa's Streamlit app, reused as-is.
- Embeddings are computed once per model and cached to disk.
- **Comparison report:**
  - outlier rate by store category (do dollar stores fit Raleigh's topics?);
  - topic coherence and diversity;
  - share of reviews in valid topics;
  - accuracy on the gold set;
  - agreement of store scores between the arms.

### Phase 5: Store scores statewide (2–3 days)
- Teresa's aggregation plus the bug fixes (store name, SE floor).
- The PyMC model is upgraded for 4,495 stores:
  - non-centred parameterisation;
  - the fast `nutpie` sampler;
  - levels: store → county → state;
  - a store-category effect, so dollar stores aren't shrunk toward a supermarket average.
- Outputs:
  - z-scores statewide, within county and within category;
  - 95% intervals.
- Run for both arms.

### Phase 6: Neighbourhoods + equity analysis (3–4 days)
- Inputs:
  - NC 2020 census tracts (~2,600);
  - ACS 5-year data: income, % Black, population, and also % no-vehicle households and SNAP
    participation;
  - USDA RUCA urban/rural codes.
- Tract scores via her distance-decay formula (h = 1).
- Replicate her WLS models. Then extend them:
  - rural vs urban;
  - county fixed effects;
  - comparison to the USDA Food Access Research Atlas food-desert flags ("Are aspatial deserts
    different from spatial ones?").

### Phase 7: Outputs (2 days)
- **Tables:** store, tract and county score tables (CSV).
- **Maps:** per-dimension statewide maps plus an interactive map.
- **Write-ups:**
  - a methods note covering what changed from Raleigh and why;
  - a frozen-vs-NC arm comparison.

---

## Part 8. Ideas for improvement (beyond replication), ranked by value ÷ effort

1. **Gold-standard labelled set with two raters** (in Phase 3). This fixes the weakest part of the
   thesis, validation on n = 9 topics, and enables every comparison below.
2. **Dimension-specific sentiment instead of stars.** A 2★ review that says "prices are great but the
   staff were rude" currently counts as neutral for *both* dimensions. Score sentiment per dimension
   from the text instead: aspect-based sentiment via an NLI zero-shot classifier, or an LLM.
3. **Review-level labels instead of topic-inherited labels.** Today every review in "Meat Department
   Quality" gets the same label. Compare each review (or each sentence) to the definitions directly,
   and use the topic only as a prior.
4. **An LLM as annotator** (Claude or an open model like Qwen3). Have it label the gold set and a
   large sample with the 5 A's plus sentiment. Measure its agreement with humans. If it's high, use
   it on the full corpus (batch processing) as a third arm.
5. **Better likelihood in the pooling model.** The signal takes only 3 values (−1/0/+1), so an
   ordinal or multinomial model is more correct than a normal approximation. Also borrow strength
   across the 5 dimensions (a multivariate model).
6. **Real travel times instead of straight lines:**
   - OSRM/OpenRouteService drive and walk times;
   - GTFS transit where available (Raleigh, Charlotte, Durham…);
   - calibrate the decay parameter instead of fixing h = 1.
7. **Foot-traffic data** (`Dr.ben_data`, Advan/SafeGraph patterns). Weight stores by where their
   visitors actually come from, instead of assuming people use nearby stores. This is Teresa's own
   stated future work, and we already have the data.
8. **Time:**
   - approximate review dates enable trends: before vs after COVID, or a store getting worse;
   - recency weighting;
   - "when are food deserts" (Widener).
9. **Bias diagnostics:**
   - review volume per capita by tract;
   - Local Guide vs casual reviewers;
   - duplicate or near-duplicate (possibly fake) reviews;
   - reviews per store vs the store's foot traffic.
10. **Reproducibility:**
    - scripts and config instead of notebooks with `G:\` paths;
    - fixed seeds;
    - cached embeddings with hashes;
    - one command per phase.

### Open questions for Teresa / Dr. Rachunok
- Which definition text produced the thesis results: first-person (slides) or third-person (`.json`)?
- What is the exact distance-decay function for h = 1 (exponential or power)?
- How were the 18 invalid topics decided? Was the scorer the same person as in the demo pipeline?
- Is star-based sentiment the intended final design? The SRA deck mentions "similarity-weighted
  dimension scores", and the backup slides have a VADER/TextBlob sentiment validation.
- Scope: should Walmart/Target supercenters and dollar stores be compared on the same scale as
  supermarkets, or within their category only?

---

## Part 9. Learn the background: watch and read list

**Start here:** about 2 hours of video gives you everything in Part 3.

| Topic | Resource |
|---|---|
| What a neural network is | 3Blue1Brown, *"But what is a neural network?"*: https://www.youtube.com/watch?v=aircAruvnKk |
| Transformers, visually | 3Blue1Brown, *"Transformers, the tech behind LLMs"* (DL ch. 5): https://www.youtube.com/watch?v=wjZofJX0v4M |
| Attention | 3Blue1Brown, *"Attention in transformers, step-by-step"* (DL ch. 6): https://www.youtube.com/watch?v=eMlx5fFNoYc |
| LLMs for non-experts | Andrej Karpathy, *"[1hr Talk] Intro to Large Language Models"*: https://www.youtube.com/watch?v=zjkBMFhNj_g |
| Word embeddings | StatQuest, *"Word Embedding and Word2Vec, Clearly Explained!!!"* (search the title on YouTube) |
| Cosine similarity | StatQuest, *"Cosine Similarity, Clearly Explained!!!"* (search the title) |
| UMAP | StatQuest, *"UMAP Dimension Reduction, Main Ideas!!!"* (search the title) |
| HDBSCAN | John Healy, PyData NYC 2018, *"HDBSCAN, Fast Density Based Clustering, the How and the Why"* (search the title) |
| BERTopic from its author | *"BERTopic for Topic Modeling – Maarten Grootendorst – Talking Language AI Ep#1"* (Cohere; search the title) |
| Bayesian multilevel models / partial pooling | Richard McElreath, *Statistical Rethinking 2023*, lectures on multilevel models (YouTube playlist "Statistical Rethinking 2023") |

**Read:**
- Jay Alammar, *The Illustrated Transformer* and *The Illustrated Word2vec* (jalammar.github.io).
- Alammar & Grootendorst, *Hands-On Large Language Models* (O'Reilly 2024). Chapter 4 (text
  classification) and chapter 5 (text clustering and topic modelling) are almost exactly Teresa's
  pipeline.
- Docs:
  - sentence-transformers: sbert.net;
  - BERTopic: maartengr.github.io/BERTopic (see "Algorithm" and "Zero-shot topic modeling").
- **Papers in `Teresa Files/`:**
  - read the **thesis PDF** first;
  - then Widener (when are food deserts);
  - then Konapur (5 A's);
  - then Caspi and Turner (reviews of measures);
  - then Sawyer (systems view).
- **Leaderboard:** MTEB (huggingface.co/spaces/mteb/leaderboard). Use it to shortlist embedding
  models, then test on our own gold set.

---

## Verification (end to end)
- **Phase 1:** `test_export.py` covers counts, uniqueness, coordinates and language coverage.
- **Phase 2:** Raleigh reproduction ρ ≥ 0.8 per dimension on the 42 matched stores.
- **Phase 3:** gold-set kappa reported. Bake-off table with accuracy and F1 for each model.
- **Phase 4:** comparison report for both arms; outlier rates by category.
- **Phase 5:** r-hat ≤ 1.01 and ESS > 400 for all parameters. A shrinkage plot shows small-n stores
  pulled in.
- **Phase 6:** the Raleigh-only regression approximately reproduces her coefficient signs (slides 56–59).

## Critical files
- **Method:** `Teresa Project/Code Pipeline/{2,3,6}_*.ipynb`, `apps/manual_validation_app.py`
- **Frozen model:** `Teresa Project/Raleigh_Data/final_embedding_model/`, `definition_embeddings/`
- **Ground truth:** `Teresa Project/Raleigh_Data/{store_scores.csv, raleigh_ring_reviews_with_dims.csv}`
- **NC data** (read-only):
  - `Gmaps Scraper/Data/Step_5_master_reviews.db`
  - `Gmaps Scraper/Frame/build/probe.db` (county)
  - `Gmaps Scraper/Dr.ben_data/dr_ben_patterns.db` (foot traffic)
- **Export spec:** `Gmaps Scraper/STATUS.md` §4.4 and §6 step 9

## After approval
- Publish this guide as a shareable document (Claude Doc) so others on the team can read it.
- Then start Phase 0, pausing for review at the end of each phase.

**Sources for the model landscape:**
- https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting
- https://openrouter.ai/blog/insights/best-embedding-models-2026/
- https://www.datallmlab.com/blog/best-embedding-models.html
