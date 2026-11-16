"""Paths and parameters for the NC scale-up of Teresa Groton's review pipeline.

Steps (Teresa's method, see "Teresa Project/Teresa Files/Aspatial_Access_NC_Plan.md"):
  1  embed each review                         step1_embed.py
  2  reduce the embeddings with UMAP           step2_umap.py
  3  cluster into topics (HDBSCAN, BERTopic)   step3_topics.py
  4  match topics to the five access dimensions (not built yet)
  5  pool reviews into store scores             (not built yet)
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # aspatial_food_access_research/
DATA = HERE / "data"                                 # all outputs of this pipeline

# Input: the NC master review DB, opened read-only. Its counts equal freeze v2
# (Step_5_master_reviews_FROZEN_2026-10-24.db: 4,442 stores, 1,941,538 reviews, 977,398 with text).
REVIEWS_DB = ROOT / "Reviews Data" / "Step_5_master_reviews.db"

# Teresa's Raleigh pilot (the frozen-model arm and her parameters).
TERESA = ROOT / "Teresa Project" / "Code Pipeline"
RALEIGH_MODEL = TERESA / "bertopic_model"

# ---- step 1: cleaning + embedding (Teresa notebook 2, cells 6-10) ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
EMBED_BATCH = 512
# Teresa's list, verbatim (Raleigh store names).
TERESA_STOPWORDS = ['publix', 'harris', 'teeter', 'ht', 'teeters', 'Publix', 'Harris', 'Teeter', 'HT', 'lion', 'Lion', 'lions',
                    'aldi', 'Aldi', 'aldis', 'lidl', 'Lidl', 'Carlie', 'carlie', 'wegmans', 'Wegmans', 'sprouts', 'lowes', 'Lowes', 'trader', 'Trader',
                    'joes', 'Joes', 'walmart', 'Walmart', 'kroger', "Kroger", 'Aldis',
                    'piggly', 'Piggly', 'piggly wiggly', 'Piggly Wiggly', 'pigglywiggly', 'PigglyWiggly', 'wiggly', 'piggy']
# NC additions: distinctive tokens of chains with >5 NC stores. Brand words that are ordinary
# English (dollar, general, family, tree, target, fresh, market, foods, whole, club, outlet, big,
# lots, vitamin) are kept: removing them would erase real content ("dollar" = affordability).
NC_STOPWORDS = ['ingles', 'iga', 'gnc', 'costco', 'costcos', 'sams', 'dg', 'carlies', 'shoppe',
                'walmarts', 'krogers', 'publixs', 'lidls', 'wegman']
STOPWORDS = TERESA_STOPWORDS + NC_STOPWORDS

# ---- step 2: UMAP (Teresa notebook 2, cell 14) ----
UMAP_PARAMS = dict(n_neighbors=30, n_components=5, min_dist=0.05, metric="cosine", random_state=42)
UMAP_VIZ_PARAMS = dict(n_neighbors=15, n_components=2, min_dist=0.05, metric="cosine", random_state=42)  # cell 11
# "cuml" = RAPIDS GPU UMAP / HDBSCAN (BERTopic's documented GPU path; minutes, little host RAM),
# "cpu" = umap-learn / hdbscan as in Teresa's notebook (needs ~6 GB free RAM and ~1 h at NC scale).
ENGINE = "cuml"

# ---- step 3: HDBSCAN + BERTopic (cells 16, 18, 22-29) ----
# Teresa: min_cluster_size=120 on 32,751 Raleigh reviews (59 topics). NC has ~30x more reviews,
# so the cluster size is re-tuned on a grid (step3_topics.py grid) and the choice recorded here.
HDBSCAN_FIXED = dict(min_samples=15, metric="euclidean", cluster_selection_method="leaf", prediction_data=False)
MIN_CLUSTER_GRID = [250, 500, 1000, 2000, 4000, 8000]
MIN_CLUSTER_SIZE = 2000            # chosen 2026-10-25 from the grid (README.md, "Choosing the cluster size")
VECTORIZER_PARAMS = dict(stop_words="english", min_df=2, ngram_range=(1, 3))
MMR_DIVERSITY = 0.3
TOP_N_WORDS = 10
COHERENCE_SAMPLE = 100_000         # docs used for C_v coherence (full corpus is too slow)
SEED = 42

# ---- experiment: reviews with no specific words (exp_no_empty.py) ----
# Praise, filler and shopping words. A review made only of these and English stop words says nothing
# about which part of access it concerns. Hand-made list; not used by the shipped steps 1-3.
GENERIC_WORDS = frozenset("""good great nice love loved loves like best better awesome excellent amazing
wonderful fantastic perfect fine ok okay cool super pretty really very always just place store stores
shop shops shopping go going get got went come came people lol wow yes thanks thank everything everyone
time times one thing things well much lot lots little big bad worst terrible horrible hate poor sucks
decent average favorite recommend highly experience visit overall nothing stars star""".split())

# ---- step 4: topic -> dimension matching (Teresa notebook 3, cells 22-31) ----
# Definition text: "first_person" (thesis slide 71) reproduces her saved Raleigh topic probabilities to within
# 0.0016 (decisions/d1_definitions.py); "third_person" (her demo folder's definition_text.json) does not.
DEFINITIONS = "first_person"
SOFTMAX_TAU = 0.11
MARGIN_DELTA = 0.17

# ---- step 6: store scores (Teresa notebook 6) ----
SIGNAL = {1: -1, 2: 0, 3: 0, 4: 0, 5: 1}
SE_FLOOR = 1e-6                    # her clip; keeps 1-review stores unshrunk (a known issue, fixed only in Pipeline B)
POOL_DRAWS, POOL_TUNE, POOL_CHAINS, POOL_TARGET_ACCEPT, POOL_SEED = 2000, 2000, 4, 0.95, 123
# Validity: data/step5_manual_scores.csv from the human check. Until it exists, topics drafted as general
# praise / no clear subject in parents_draft.csv are treated as invalid (marked "provisional" in the outputs).

# ---- step 7: neighbourhoods (thesis eq. 4.19-4.21) ----
DECAY_H_MILES = 1.0                # w = exp(-D / h), h = 1 mile (USDA urban definition)
CENSUS = HERE / "data_external" / "census"
