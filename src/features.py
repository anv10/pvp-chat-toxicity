"""Feature extractors for the classic ML models.

- word TF-IDF:  what the paper used. Each message becomes a sparse vector of
                word (and word-pair) weights.
- char TF-IDF:  our addition. Character n-grams ("noo", "oob", "f*g") are
                robust to gamer spelling and obfuscation, where whole-word
                features fail on every new variant.
- lexicon counts: the paper's Naive Bayes baseline. One column per toxic
                phrase in our lexicons, holding how often it occurs.
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.labeling import Lexicon, load_lexicon


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        # Keep 1-letter tokens: "u" in "u suck" matters in chat.
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),   # bigrams capture "gg ez", "my bad", "you're bad"
        min_df=2,             # drop words seen once: mostly typos/noise
        sublinear_tf=True,    # 1 + log(tf): a word repeated 5x isn't 5x as toxic
    )


def char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",   # n-grams inside word boundaries
        ngram_range=(2, 5),
        min_df=3,
        max_features=100_000,
        sublinear_tf=True,
    )


def word_char_tfidf() -> FeatureUnion:
    """Word and char features side by side (concatenated)."""
    return FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])


class LexiconCounter(BaseEstimator, TransformerMixin):
    """Paper's NB baseline features: count of each dictionary phrase per message.

    Deliberately context-free (no "my bad" overrides), as in the paper, which is
    exactly why it struggles with words like "bad" that depend on context.
    """

    def fit(self, X, y=None):
        self.terms_ = load_lexicon("derogatory.txt") + load_lexicon("negative.txt")
        self.lexicon_ = Lexicon(self.terms_)
        self.index_ = {t: i for i, t in enumerate(self.terms_)}
        return self

    def transform(self, X):
        rows, cols = [], []
        for r, text in enumerate(X):
            for term in self.lexicon_.findall(text.replace("'", "")):
                rows.append(r)
                cols.append(self.index_[term])
        data = np.ones(len(rows), dtype=np.float64)
        # Duplicate (row, col) pairs are summed, giving counts.
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(X), len(self.terms_)))

    def get_feature_names_out(self, input_features=None):
        return np.array(self.terms_)


FEATURES = {
    "word": word_tfidf,
    "char": char_tfidf,
    "word+char": word_char_tfidf,
    "lexicon": LexiconCounter,
}
