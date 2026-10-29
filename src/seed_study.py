"""Re-train the models that use randomness with several seeds, to see how much
their scores move from run to run (dedup data only, nothing is saved but the table).

Run:  python -m src.seed_study                 # seeds 42-46
      python -m src.seed_study --seeds 42,43 --only lstm_gameft100

Covers the random forests and all 8 neural models. SVM, Naive Bayes, KNN and
logistic regression are deterministic given the data, so a new seed changes nothing.
Output: results/seed_study.csv (one row per model x seed)
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from src.config import RESULTS_DIR, set_seeds
from src.embeddings import EMBEDDINGS, load_embedding
from src.nn_models import build_model, build_vocab, embedding_matrix, encode
from src.train_ml import MODEL_SPECS, build, load_split
from src.train_nn import ARCHS, BATCH_SIZE, MAX_EPOCHS, PATIENCE

RF_KEYS = ["rf_gini", "rf_entropy", "rf_entropy_d32"]
NN_KEYS = [f"{a}_{e}" for a in ARCHS for e in EMBEDDINGS]


def metrics(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2])}


def row(key, name, seed, val, test, secs, y_val, y_test) -> dict:
    v, t = metrics(y_val, val), metrics(y_test, test)
    return {"key": key, "model": name, "seed": seed, "val_accuracy": v["accuracy"],
            "test_accuracy": t["accuracy"], "test_macro_f1": t["macro_f1"], "seconds": round(secs)}


def run_rf(key, seed, tr, va, te) -> dict:
    pipe, (name, *_rest) = build(key)
    pipe.set_params(clf__random_state=seed)
    t0 = time.perf_counter()
    pipe.fit(tr["text_norm"].tolist(), tr["label"].to_numpy())
    secs = time.perf_counter() - t0
    return row(key, name, seed, pipe.predict(va["text_norm"].tolist()),
               pipe.predict(te["text_norm"].tolist()), secs, va["label"], te["label"])


def run_nn(key, seed, tr, va, te, kv) -> dict:
    from tensorflow import keras

    arch, emb = key.split("_", 1)
    set_seeds(seed)
    keras.utils.set_random_seed(seed)
    vocab = build_vocab(tr["text_norm"], kv)
    X = {n: encode(d["text_norm"], vocab) for n, d in (("tr", tr), ("va", va), ("te", te))}
    model = build_model(arch, embedding_matrix(vocab, kv))
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                         restore_best_weights=True)
    t0 = time.perf_counter()
    model.fit(X["tr"], tr["label"].to_numpy(), validation_data=(X["va"], va["label"].to_numpy()),
              epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[stop], verbose=0)
    secs = time.perf_counter() - t0
    name = f"{ARCHS[arch]} ({emb})"
    return row(key, name, seed, model.predict(X["va"], verbose=0).argmax(1),
               model.predict(X["te"], verbose=0).argmax(1), secs, va["label"], te["label"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--only", help="comma-separated model keys")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    keys = args.only.split(",") if args.only else RF_KEYS + NN_KEYS
    tr, va, te = (load_split("dedup", s) for s in ("train", "val", "test"))

    rows = []
    for key in keys:
        if key in RF_KEYS:
            for seed in seeds:
                rows.append(run_rf(key, seed, tr, va, te))
                print(rows[-1], flush=True)
    for emb in EMBEDDINGS:
        emb_keys = [k for k in keys if k in NN_KEYS and k.split("_", 1)[1] == emb]
        if not emb_keys:
            continue
        kv = load_embedding(emb)
        for key in emb_keys:
            for seed in seeds:
                rows.append(run_nn(key, seed, tr, va, te, kv))
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "seed_study.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "seed_study.csv", index=False)
    g = pd.DataFrame(rows).groupby("model")[["test_accuracy", "test_macro_f1"]].agg(["mean", "std"])
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
