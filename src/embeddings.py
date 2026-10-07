"""Word embeddings for the neural models: pretrained GloVe vs our own game-chat word2vec.

Run:  python -m src.embeddings     # train our word2vec + write the comparison report

The paper concluded that embeddings pretrained on news text don't understand
gamer slang ("feeder", "bronze", "gg"). Our extension tests that directly:
  - glove50 / glove100      pretrained on Wikipedia + Gigaword news (via gensim downloader)
  - gamew2v50 / gamew2v100  word2vec trained by us on 1.8M game chat messages
                            (data/processed/w2v_corpus.txt; val/test messages excluded)
Besides model accuracy, we compare (a) how much of our chat vocabulary each
embedding knows and (b) nearest neighbours of slang words.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec

from src.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from src.text_utils import tokenize

EMBEDDINGS = {
    "glove50": "glove-wiki-gigaword-50",
    "glove100": "glove-wiki-gigaword-100",
    "gamew2v50": 50,
    "gamew2v100": 100,
}
W2V_DIR = MODELS_DIR / "word2vec"
SLANG_PROBES = ["noob", "feeder", "gg", "ez", "bronze", "report", "jungle", "ff", "carry", "toxic"]


def stable_hash(word: str) -> int:
    """gensim seeds each word's initial vector with hash(word). Python's built-in
    hash() changes between runs, so we use a fixed one for reproducible vectors."""
    return int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)


class ChatCorpus:
    """Streams tokenised messages from disk, so 1.8M lines never sit in RAM as lists."""

    def __iter__(self):
        with open(PROCESSED_DIR / "w2v_corpus.txt", encoding="utf-8") as f:
            for line in f:
                tokens = tokenize(line)
                if tokens:
                    yield tokens


def train_game_w2v(dim: int) -> KeyedVectors:
    """Skip-gram word2vec on game chat. workers=1 so the result is reproducible
    (gensim with several threads is not deterministic even with a fixed seed)."""
    model = Word2Vec(
        sentences=ChatCorpus(),
        vector_size=dim,
        window=5,       # chat messages are short; 5 words of context is most of a message
        min_count=5,    # ignore words seen < 5 times (typos, gibberish)
        sg=1,           # skip-gram: better than CBOW for rare words, i.e. slang
        epochs=5,
        workers=1,
        seed=SEED,
        hashfxn=stable_hash,
    )
    W2V_DIR.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(W2V_DIR / f"gamew2v{dim}.kv"))
    return model.wv


def load_embedding(name: str) -> KeyedVectors:
    source = EMBEDDINGS[name]
    if isinstance(source, str):
        import gensim.downloader as api  # downloads once, then cached in ~/gensim-data
        return api.load(source)
    path = W2V_DIR / f"{name}.kv"
    if not path.exists():
        print(f"  training {name} on game chat (a few minutes) ...")
        return train_game_w2v(source)
    return KeyedVectors.load(str(path))


def coverage(kv: KeyedVectors, train_tokens: list[list[str]]) -> dict:
    """Share of our training vocabulary (types) and running text (tokens) the embedding knows."""
    counts = pd.Series([t for msg in train_tokens for t in msg]).value_counts()
    known = counts.index.map(lambda w: w in kv.key_to_index).to_numpy(dtype=bool)
    return {"vocab_coverage": known.mean(), "token_coverage": counts[known].sum() / counts.sum()}


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv", keep_default_na=False)
    train_tokens = [tokenize(t) for t in train["text_norm"]]
    rows, neighbours = [], {}
    for name in EMBEDDINGS:
        kv = load_embedding(name)
        rows.append({"embedding": name, "dim": kv.vector_size, "known_words": len(kv),
                     **coverage(kv, train_tokens)})
        neighbours[name] = {
            w: ([n for n, _ in kv.most_similar(w, topn=6)] if w in kv.key_to_index else None)
            for w in SLANG_PROBES}
    cov = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    cov.round(4).to_csv(RESULTS_DIR / "embedding_coverage.csv", index=False)
    (RESULTS_DIR / "embedding_neighbours.json").write_text(json.dumps(neighbours, indent=2))

    lines = ["# GloVe (news) vs our game-chat word2vec", "",
             "## Coverage of our training messages", "",
             "| Embedding | Dim | Words known | Vocab coverage | Token coverage |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['embedding']} | {r['dim']} | {r['known_words']:,} "
                     f"| {r['vocab_coverage']:.1%} | {r['token_coverage']:.1%} |")
    lines += ["", "## Nearest neighbours of gaming slang (50d versions)", "",
              "| Word | GloVe 50d | Game-chat word2vec 50d |", "|---|---|---|"]
    for w in SLANG_PROBES:
        g = neighbours["glove50"][w]
        m = neighbours["gamew2v50"][w]
        lines.append(f"| {w} | {', '.join(g) if g else '(unknown)'} "
                     f"| {', '.join(m) if m else '(unknown)'} |")
    (RESULTS_DIR / "embeddings.md").write_text("\n".join(lines) + "\n")
    print(cov.round(3).to_string(index=False))
    print("\nWrote results/embeddings.md")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
