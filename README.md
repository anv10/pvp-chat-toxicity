# Trolls, Haters, or Teammates? Classifying toxicity in PvP game chat

A replication and extension of the Stanford CS229 project
[*Trolls, Haters, or Teammates: Classifying Sentiment in the Language of PvP Game Communications*](https://cs229.stanford.edu/proj2021spr/report2/81982391.pdf)
(Borst, Yang, Zhu, 2021).

Each in-game chat message is assigned to one of three classes:

| Class | Meaning | Examples |
|---|---|---|
| **0** | neutral / positive | "ggwp", "thanks", "nice", "my bad" |
| **1** | negative attitude | "noob team", "such a feeder", "report this useless guy", "gg ez" |
| **2** | derogatory | slurs, sexist / racist / homophobic / ableist insults, "kys" |

> **Content warning:** the lexicons, error dumps and EDA plots contain offensive language. That is unavoidable for a toxicity classifier.

## Key findings

1. **Duplicates explain the paper's 99% scores.** About 58% of English game-chat messages are repeats ("gg", "ez", ...). If duplicates are not removed before splitting, **31.6% of test messages also appear word-for-word in train**. On those messages every model scores 92–100%, which is memorisation. On genuinely new messages, KNN (K=1) drops from the paper's 98.5% to **67%**.
2. **On clean (deduplicated) data, linear models with character n-grams win.** The best 3-class model is Logistic Regression on word + char TF-IDF, with **98.4% accuracy / 0.984 macro F1**. The best binary model is a linear SVM on char n-grams (98.7%).
3. **Our extension: game-chat word2vec beats GloVe in all four comparisons** (feedforward and LSTM, 50d and 100d), by +1.6 to +2.3 accuracy points. GloVe covers only 56% of our chat vocabulary, against 74% for ours. In GloVe, "feeder" sits next to railway words and "bronze" next to "medal". In ours they sit next to "flamer"/"noob" and "platinum"/"diamond". This directly supports the paper's conclusion that news embeddings don't understand gamer slang.
4. **Classic ML still beats our neural nets** (best LSTM: 93.9%), matching the paper's finding. Word-level embeddings can't see obfuscated spellings that char n-grams catch.

![Our accuracy vs the paper's](results/figures/paper_comparison.png)

## Results

All numbers are on the held-out **test set (6,600 messages)** of the deduplicated data.
**Labels are weak labels from our lexicon rules** (see [Labeling](#how-labels-were-made)), so scores
measure agreement with those rules. The full table, with per-class F1 and validation scores, is in
[results/summary.md](results/summary.md) and [results/summary.csv](results/summary.csv).

| Model | Task | Features | Accuracy | Macro F1 | Paper accuracy |
|---|---|---|---:|---:|---:|
| SVM, linear kernel | binary (0 vs 1+2) | word TF-IDF | 0.970 | 0.970 | 0.9565 |
| SVM, RBF kernel | binary | word TF-IDF | 0.961 | 0.961 | 0.9635 |
| SVM, polynomial kernel | binary | word TF-IDF | 0.875 | 0.875 | 0.9435 |
| *SVM, linear kernel (ours)* | binary | char TF-IDF | **0.987** | **0.986** | – |
| Naive Bayes, lexicon dictionary | 3-class | lexicon counts | 0.927 | 0.927 | 0.8500 |
| Naive Bayes | 3-class | word TF-IDF | 0.877 | 0.872 | 0.9506 |
| *Naive Bayes (ours)* | 3-class | char TF-IDF | 0.906 | 0.905 | – |
| KNN, K=1 | 3-class | word TF-IDF | 0.644 | 0.619 | 0.9853 |
| KNN, K=2 | 3-class | word TF-IDF | 0.632 | 0.577 | 0.9711 |
| KNN, K=3 | 3-class | word TF-IDF | 0.642 | 0.597 | 0.9683 |
| KNN, K=6 | 3-class | word TF-IDF | 0.596 | 0.516 | 0.8424 |
| Random Forest, Gini | 3-class | word TF-IDF | 0.952 | 0.949 | 0.9912 |
| Random Forest, entropy | 3-class | word TF-IDF | 0.949 | 0.946 | 0.9910 |
| Random Forest, entropy, max_depth=32 | 3-class | word TF-IDF | 0.779 | 0.764 | 0.8775 |
| *Logistic Regression (ours)* | 3-class | word + char TF-IDF | **0.984** | **0.984** | – |
| Feedforward NN (ReLU, ReLU, softmax) | 3-class | GloVe 50d | 0.820 | 0.812 | 0.9486 |
| Feedforward NN | 3-class | GloVe 100d | 0.857 | 0.852 | – |
| *Feedforward NN (ours)* | 3-class | game-chat word2vec 50d | 0.843 | 0.829 | – |
| *Feedforward NN (ours)* | 3-class | game-chat word2vec 100d | 0.880 | 0.870 | – |
| LSTM | 3-class | GloVe 50d | 0.918 | 0.917 | 0.9586 |
| LSTM | 3-class | GloVe 100d | 0.921 | 0.918 | 0.9175 |
| *LSTM (ours)* | 3-class | game-chat word2vec 50d | 0.934 | 0.930 | – |
| *LSTM (ours)* | 3-class | game-chat word2vec 100d | **0.939** | **0.936** | – |

Rows in *italics* are our additions. The paper's Google-News word2vec models were skipped (1.6 GB download) and replaced by our own game-chat word2vec.

### Side experiment: what duplicates do

Same models, same size and class balance, but sampled **without** deduplication:

| Model | Accuracy, dedup | Accuracy, with dups | Accuracy on test messages *seen* in train | Accuracy on *unseen* messages |
|---|---:|---:|---:|---:|
| KNN, K=1 | 0.644 | 0.775 | 0.998 | 0.673 |
| Random Forest, entropy, depth 32 | 0.779 | 0.863 | 0.916 | 0.838 |
| Feedforward NN, GloVe 50d | 0.820 | 0.873 | 0.969 | 0.829 |
| Random Forest, Gini | 0.952 | 0.963 | 0.998 | 0.947 |
| Logistic Regression, word+char | 0.984 | 0.987 | 0.999 | 0.982 |

Every model's score goes up with duplicates; for KNN the gain is +13 points. All 23 models are in [results/summary.md](results/summary.md) and charted in [leakage_effect.png](results/figures/leakage_effect.png).

### Extension: GloVe vs our own word2vec

| Embedding | Words known | Our vocabulary covered | Our running text covered |
|---|---:|---:|---:|
| GloVe (Wikipedia + news) | 400,000 | 56.1% | 90.3% |
| Game-chat word2vec (ours) | 38,141 | 74.3% | 96.8% |

| Word | Nearest neighbours in GloVe 50d | Nearest neighbours in our word2vec 50d |
|---|---|---|
| feeder | overground, caledonian, mainline | flamer, nob, noob |
| bronze | silver, gold, medal | silver, platin, platinum, diamond |
| ff | trelleborgs, malmö, viborg | end, surr, surender |
| toxic | radioactive, poisonous, wastes | unskilled, annoying, childish |

More neighbours are in [results/embeddings.md](results/embeddings.md).

## Data

Both datasets are public on Kaggle and must be **downloaded manually** into `data/raw/` (gitignored, ~940 MB):

| File | Dataset | Size used |
|---|---|---|
| `data/raw/chatlogs.csv` | [League of Legends Tribunal chatlogs](https://www.kaggle.com/datasets/simshengxue/league-of-legends-tribunal-chatlogs) (simshengxue) | all 1,697,222 messages |
| `data/raw/dota2_chat_messages.csv` | [GOSU.AI Dota 2 game chats](https://www.kaggle.com/datasets/romovpa/gosuai-dota-2-game-chats) (romovpa) | first 5,000,000 of 21.7M messages |

Neither dataset has message-level toxicity labels. The LoL report reasons apply to a whole game, not to a single message.

| Pipeline step | Messages |
|---|---:|
| Raw (LoL + first 5M Dota) | 6,697,222 |
| English-ish, non-empty, non-system | 4,456,960 |
| Unique after normalisation + deduplication | 1,851,646 (58.5% were duplicates) |
| Final balanced sample | **33,000**: 45% neutral / 30% negative / 25% derogatory |
| Train / validation / test (stratified, seed 42) | 21,120 / 5,280 / 6,600 |

### How labels were made

We used weak labeling ([src/labeling.py](src/labeling.py)), because hand-labeling 33k messages wasn't possible in one week. The rules are applied in this order:

1. **Neutral overrides** are removed first: "my bad", "i suck", "not bad", "dont report", ...
2. **Derogatory lexicon** match → class 2. Obfuscations are handled: leetspeak (`f4g`, `n1gg3r`), masking (`f*g`), stretched letters (`faaag`), separators (`f.a.g`) and spaced letters (`f a g`).
3. **Negative lexicon** match, or a context rule ("you're bad" / "bad jungle", but not "my bad") → class 1.
4. Otherwise → class 0. "gg" alone is neutral, while "gg ez" is negative because of "ez".

The word lists are plain text files you can edit in [data/lexicons/](data/lexicons/). [tests/test_labeling.py](tests/test_labeling.py) has 64 test cases, including false-positive traps like "therapist", "grape" and "ezreal".

**Label quality check:** [data/manual_check.csv](data/manual_check.csv) is a random sample of 300 messages for hand labeling. Fill in `manual_label` (0/1/2), hiding the `auto_label` column while you label, then run `python -m src.label_agreement`. It reports accuracy and Cohen's kappa between the rules and a human.

## Setup

Requires Python 3.13 (tested on macOS, CPU only; no GPU needed).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then download the two Kaggle CSVs into `data/raw/` (see [Data](#data)).

## How to run

Run every command from the project root with the virtual environment active. Times are on a laptop CPU.

```bash
python -m src.prepare_data     # ~6 min  clean, dedup, weak-label, sample 33k, split
python -m src.train_ml         # ~4 min  15 classic models x {dedup, with_dups}
python -m src.train_nn --dataset both
                               # ~10 min 8 neural models x 2 datasets; first run downloads
                               #         GloVe (~190 MB) and trains our word2vec
python -m src.embeddings       # ~1 min  GloVe vs word2vec coverage + neighbour report
python -m src.evaluate         # ~2 min  metrics, confusion matrices, summary tables
python -m src.error_analysis   #         misclassified test messages for the best model
streamlit run app/streamlit_app.py   # live demo at http://localhost:8501
```

Optional:

```bash
python -m src.label_agreement          # after filling in data/manual_check.csv
python -m src.error_analysis --all     # error dumps for every model
python -m pytest tests/                # labeler sanity tests
# EDA: open notebooks/01_eda.ipynb in VS Code (or re-run it headless:)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

### Demo

The Streamlit app classifies a typed chat message. It shows the predicted class, a probability chart, a model selector (all 23 models, sorted by test macro F1), the label the lexicon rules would give, and a "compare all models" view. The default model (Logistic Regression) predicts in about 0.02 s, and an LSTM in about 0.2 s.

## Project structure

```
data/
  raw/                  Kaggle CSVs (gitignored, download manually)
  lexicons/             editable word lists: derogatory, negative, neutral overrides, non-English markers
  processed/            train/val/test splits, with_dups/ splits, stats.json
  manual_check.csv      300 messages for hand-labeling
src/
  config.py             paths, seed, class balance, split sizes
  text_utils.py         normalisation, English filter, tokenizer
  labeling.py           weak labeler (lexicons + obfuscation-aware regex + context rules)
  prepare_data.py       raw -> clean -> dedup -> label -> sample -> split
  features.py           word / char TF-IDF and lexicon-count features
  train_ml.py           SVM, Naive Bayes, KNN, Random Forest, Logistic Regression
  embeddings.py         GloVe loading, our game-chat word2vec, coverage report
  nn_models.py          feedforward NN + LSTM (Keras) and a text-in wrapper
  train_nn.py           neural training with early stopping
  model_io.py           one save/load format for every model
  evaluate.py           metrics, confusion matrices, summary tables, charts
  error_analysis.py     misclassified examples per (true, predicted) class pair
  label_agreement.py    weak labels vs hand labels (accuracy, Cohen's kappa)
  plot_style.py         shared colours and chart style
notebooks/01_eda.ipynb  class balance, message length, top words per class, duplicates
app/streamlit_app.py    demo
models/                 trained models (gitignored, ~370 MB, regenerate with the scripts)
results/                summary tables, metrics, figures, error dumps, paper numbers
tests/                  labeler and English-filter tests
```

## Design decisions

- **Deduplicate before splitting.** Text is normalised (lowercased, "noooob" → "noob", URLs and extra whitespace removed) and then deduplicated, so no normalised message appears in more than one split.
- **Vectorizers live inside the scikit-learn pipelines**, so TF-IDF vocabulary and IDF weights are fitted on training text only.
- **The word2vec corpus excludes validation and test messages**, so our embeddings never see evaluation text.
- **Early stopping** (on validation loss, patience 4, best weights restored) replaces the paper's fixed 300–550 epochs. Models stopped after 12–45 epochs.
- **The embedding layers are frozen**, so the GloVe vs word2vec comparison measures what each embedding knows, not what the network re-learned.
- **Macro F1 is reported alongside accuracy.** Accuracy is dominated by the largest class.
- **Reproducibility:** fixed seed 42 everywhere, deterministic TensorFlow ops, single-threaded word2vec with a fixed hash. Re-running gives identical numbers.

## Limitations

- **Circular evaluation.** Models are trained and tested on rule-generated labels, so high scores partly mean "the model re-learned the lexicon". The 300-message manual check is the evidence of how close the rules are to human judgement. Error analysis also shows cases where the *model* is right and the *rule* is wrong (e.g. "viper rapes every mid", which the lexicon missed).
- **Obfuscation.** The rules catch `f4g` / `f*g`, but the trained models mostly don't, because such spellings are rare in training data.
- **English filter.** It is heuristic (an ASCII ratio check plus a non-English marker-word list), so some transliterated Russian and gibberish still gets through.
- **Subjective lexicon choices.** For example, "bitch", "rape" and "kys" are placed in class 2. Each is a one-line change in [data/lexicons/](data/lexicons/).
- **Dota data is capped** at the first 5M of 21.7M rows (`DOTA_MAX_ROWS` in [src/config.py](src/config.py)). That is plenty for a 33k sample.
- **Not attempted:** fine-tuning DistilBERT (the optional stretch goal).

## Reference

K. Borst, T. H. Yang, M. Zhu. *Trolls, Haters, or Teammates: Classifying Sentiment in the Language of PvP Game Communications.* CS229 Final Project, Stanford University, Spring 2021. [PDF](https://cs229.stanford.edu/proj2021spr/report2/81982391.pdf)
