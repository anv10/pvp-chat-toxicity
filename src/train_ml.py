"""Train the paper's classic ML models (plus a few extras) on TF-IDF features.

Run:  python -m src.train_ml                    # both datasets
      python -m src.train_ml --dataset dedup    # just the clean one
      python -m src.train_ml --only knn_1,rf_gini

Each model is a scikit-learn Pipeline (features -> classifier). Keeping the
vectorizer inside the pipeline means it is fitted on TRAIN text only, so the
vocabulary and IDF weights never see val/test messages.

Models are trained on two versions of the data:
  dedup      the main, leakage-free splits
  with_dups  duplicates left in (side experiment: how much does leakage inflate scores?)
Trained models go to models/<dataset>/<key>.joblib; scoring is in src/evaluate.py.
"""
import argparse
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import SEED, SPLITS_DIR, SPLITS_WITH_DUPS_DIR, set_seeds
from src.features import FEATURES
from src.model_io import DATASETS, ModelBundle, binarize, save_bundle

# key: (display name, family, features, task, classifier factory)
# Keys match the `our_key` column of results/paper_results.csv.
MODEL_SPECS = {
    # --- Baseline 1: binary SVM, neutral (0) vs toxic (1+2), as in the paper ---
    "svm_linear_bin": ("SVM linear kernel", "SVM", "word", "binary",
                       lambda: SVC(kernel="linear", random_state=SEED)),
    "svm_rbf_bin": ("SVM RBF kernel", "SVM", "word", "binary",
                    lambda: SVC(kernel="rbf", random_state=SEED)),
    "svm_poly_bin": ("SVM polynomial kernel (deg 3)", "SVM", "word", "binary",
                     lambda: SVC(kernel="poly", degree=3, random_state=SEED)),
    # ours: same linear SVM, char n-grams instead of words
    "svm_linear_bin_char": ("SVM linear kernel (char n-grams)", "SVM", "char", "binary",
                            lambda: SVC(kernel="linear", random_state=SEED)),

    # --- Baseline 2: multinomial Naive Bayes ---
    # (a) paper's fixed toxic-phrase dictionary; (b) TF-IDF
    "nb_dict": ("Naive Bayes (lexicon counts)", "Naive Bayes", "lexicon", "multiclass",
                lambda: MultinomialNB(alpha=1.0)),  # alpha=1 = Laplace smoothing, as in paper
    "nb_tfidf": ("Naive Bayes (TF-IDF)", "Naive Bayes", "word", "multiclass",
                 lambda: MultinomialNB(alpha=0.1)),
    "nb_tfidf_char": ("Naive Bayes (char TF-IDF)", "Naive Bayes", "char", "multiclass",
                      lambda: MultinomialNB(alpha=0.1)),

    # --- KNN: Euclidean distance on (L2-normalised) TF-IDF, as in the paper ---
    **{f"knn_{k}": (f"KNN (K={k})", "KNN", "word", "multiclass",
                    (lambda k=k: KNeighborsClassifier(n_neighbors=k, n_jobs=-1)))
       for k in (1, 2, 3, 6)},

    # --- Random forests: two split criteria, plus entropy with a depth cap ---
    "rf_gini": ("Random Forest (Gini)", "Random Forest", "word", "multiclass",
                lambda: RandomForestClassifier(n_estimators=200, criterion="gini",
                                               n_jobs=-1, random_state=SEED)),
    "rf_entropy": ("Random Forest (entropy)", "Random Forest", "word", "multiclass",
                   lambda: RandomForestClassifier(n_estimators=200, criterion="entropy",
                                                  n_jobs=-1, random_state=SEED)),
    "rf_entropy_d32": ("Random Forest (entropy, max_depth=32)", "Random Forest", "word",
                       "multiclass",
                       lambda: RandomForestClassifier(n_estimators=200, criterion="entropy",
                                                      max_depth=32, n_jobs=-1,
                                                      random_state=SEED)),

    # --- ours: a strong, fast linear baseline with probabilities (demo default) ---
    "logreg_wordchar": ("Logistic Regression (word+char)", "Logistic Regression",
                        "word+char", "multiclass",
                        lambda: LogisticRegression(C=10, max_iter=2000, random_state=SEED)),
}


def load_split(dataset: str, split: str) -> pd.DataFrame:
    base = SPLITS_DIR if dataset == "dedup" else SPLITS_WITH_DUPS_DIR
    return pd.read_csv(base / f"{split}.csv", keep_default_na=False)


def build(key: str) -> tuple[Pipeline, tuple]:
    name, family, feats, task, make_clf = MODEL_SPECS[key]
    pipe = Pipeline([("features", FEATURES[feats]()), ("clf", make_clf())])
    return pipe, (name, family, feats, task)


def train_all(dataset: str, keys: list[str]) -> None:
    train = load_split(dataset, "train")
    X, y = train["text_norm"].tolist(), train["label"].to_numpy()
    print(f"\n=== {dataset}: {len(train):,} training messages ===")
    for key in keys:
        set_seeds()
        pipe, (name, family, feats, task) = build(key)
        target = binarize(y) if task == "binary" else y
        t0 = time.perf_counter()
        pipe.fit(X, target)
        secs = time.perf_counter() - t0
        n_feat = len(pipe.named_steps["features"].get_feature_names_out())
        bundle = ModelBundle(key=key, name=name, family=family, features=feats, task=task,
                             model=pipe, train_seconds=round(secs, 2),
                             extra={"n_features": n_feat, "dataset": dataset})
        path = save_bundle(bundle, dataset)
        print(f"  {key:22} {secs:7.1f}s  {n_feat:>7,} features  -> {path.relative_to(path.parents[2])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--only", help="comma-separated model keys")
    args = parser.parse_args()

    keys = args.only.split(",") if args.only else list(MODEL_SPECS)
    unknown = set(keys) - set(MODEL_SPECS)
    if unknown:
        parser.error(f"unknown model keys: {sorted(unknown)}")
    datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    for ds in datasets:
        train_all(ds, keys)
    print("\nDone. Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
