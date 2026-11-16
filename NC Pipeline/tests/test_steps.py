"""Steps 1-3: Teresa's cleaning, the streaming n-gram analyzer, and the manual-BERTopic path."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step1_embed as s1  # noqa: E402
import step3_topics as s3  # noqa: E402


def test_clean_text_matches_teresa():
    assert s1.clean_text("Harris Teeter's <b>deli</b> is GREAT!!") == "deli is great"
    assert s1.clean_text("Food Lion, Ingles & IGA prices") == "food prices"        # NC brand tokens
    assert s1.clean_text("Dollar General has low prices") == "dollar general has low prices"
    assert s1.clean_text("Publix") == ""                                            # cleaned out


@pytest.mark.parametrize("doc", ["the store was clean and the staff were very friendly",
                                 "prices prices prices!! a b cc dd", "", "ok", "Fresh produce, fresh meat; fresh fish"])
def test_streaming_ngrams_equal_sklearn(doc):
    sk = CountVectorizer(stop_words="english", ngram_range=(1, 3)).build_analyzer()
    assert Counter(s3.StreamingNgrams(3)(doc)) == Counter(sk(doc))


def test_topic_diversity_and_uniqueness():
    tu, td = s3.topic_uniqueness_and_diversity({0: ["a", "b"], 1: ["b", "c"]}, topn=2)
    assert td == 0.75 and tu == {0: 0.5, 1: 0.5}


def test_manual_bertopic_keeps_labels_and_384d_topic_embeddings():
    pytest.importorskip("bertopic")
    rng = np.random.default_rng(0)
    groups = [("cashier rude slow line checkout", 0), ("fresh produce vegetables fruit", 1),
              ("parking lot crowded cart", 2)]
    docs, labels = [], []
    for text, lab in groups:
        for k in range(40):
            docs.append(f"{text} store staff {rng.choice(['today', 'weekend', 'morning'])} visit")
            labels.append(lab)
    docs += [f"ok {t} store staff visit" for t, _ in groups] * 4      # outliers mention every theme, as in real data
    labels = np.array(labels + [-1] * 12)
    emb = rng.normal(size=(len(docs), 384)).astype(np.float32)
    model, vocab = s3.fit_topics(docs, emb, labels, representations=False)
    assert (np.asarray(model.topics_) == labels).all()
    assert np.asarray(model.topic_embeddings_).shape[1] == 384
    words = {k: [w for w, _ in v] for k, v in model.get_topics().items()}
    assert "cashier" in " ".join(words[0]) and "produce" in " ".join(words[1])


def test_count_specific_only_keeps_reviews_with_real_content():
    import exp_no_empty as ex
    assert ex.count_specific("great great great") == 0
    assert ex.count_specific("love it") == 0
    assert ex.count_specific("great place to shop") == 0
    assert ex.count_specific("always easy to get in and out") == 1      # Accessibility: must be kept
    assert ex.count_specific("parking lot is full") > 0
    assert ex.is_generic_phrase("great store") and not ex.is_generic_phrase("parking lot")
