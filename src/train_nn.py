"""Train the neural models: {feedforward, LSTM} x {GloVe 50/100, game-chat word2vec 50/100}.

Run:  python -m src.train_nn                       # all 8 models, deduplicated data
      python -m src.train_nn --only lstm_glove50
      python -m src.train_nn --dataset with_dups     # leakage side experiment

Instead of the paper's fixed 300-550 epochs we use EARLY STOPPING: after each
epoch we check the loss on the validation set, stop once it hasn't improved for
PATIENCE epochs, and keep the weights from the best epoch. That avoids both
wasted training and overfitting, and is a fairer comparison between models.
"""
import argparse
import time

import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, SEED, set_seeds
from src.embeddings import EMBEDDINGS, load_embedding
from src.model_io import DATASETS, ModelBundle, model_dir, save_bundle
from src.nn_models import KerasTextClassifier, build_model, build_vocab, embedding_matrix, encode
from src.plot_style import apply_style
from src.train_ml import load_split

ARCHS = {"ff": "Feedforward NN", "lstm": "LSTM"}
EMB_NAMES = {"glove50": "GloVe 50d", "glove100": "GloVe 100d",
             "gameft50": "game-chat fastText 50d", "gameft100": "game-chat fastText 100d"}
MODEL_KEYS = [f"{a}_{e}" for a in ARCHS for e in EMBEDDINGS]

MAX_EPOCHS = 100
PATIENCE = 4
BATCH_SIZE = 64


def plot_history(history: dict, title: str, path) -> None:
    """Loss and accuracy curves per epoch (the paper's Figure 6)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    epochs = range(1, len(history["loss"]) + 1)
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(epochs, history[metric], color="#2a78d6", linewidth=2, label="train")
        ax.plot(epochs, history[f"val_{metric}"], color="#eb6834", linewidth=2, label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(metric)
    axes[0].legend()
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train_one(key: str, dataset: str, train, val, kv) -> None:
    import tensorflow as tf
    from tensorflow import keras

    arch, emb = key.split("_", 1)
    # Same seed for every model; deterministic ops so reruns give identical numbers.
    set_seeds()
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    vocab = build_vocab(train["text_norm"], kv)
    X_tr, X_va = encode(train["text_norm"], vocab), encode(val["text_norm"], vocab)
    y_tr, y_va = train["label"].to_numpy(), val["label"].to_numpy()

    model = build_model(arch, embedding_matrix(vocab, kv))
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                         restore_best_weights=True)
    t0 = time.perf_counter()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=MAX_EPOCHS,
                     batch_size=BATCH_SIZE, callbacks=[stop], verbose=0)
    secs = time.perf_counter() - t0

    model_file = f"{dataset}/{key}.keras"
    model_dir(dataset).mkdir(parents=True, exist_ok=True)
    model.save(model_dir(dataset).parent / model_file)
    name = f"{ARCHS[arch]} ({EMB_NAMES[emb]})"
    epochs_run = len(hist.history["loss"])
    best_epoch = epochs_run - PATIENCE if stop.stopped_epoch else epochs_run
    bundle = ModelBundle(
        key=key, name=name, family=ARCHS[arch], features=emb, task="multiclass",
        model=KerasTextClassifier(vocab, model_file), train_seconds=round(secs, 2),
        extra={"dataset": dataset, "epochs_run": epochs_run, "best_epoch": best_epoch,
               "vocab_size": len(vocab), "params": model.count_params()})
    save_bundle(bundle, dataset)
    plot_history(hist.history, f"{name} - {dataset}",
                 FIGURES_DIR / f"nn_history_{dataset}_{key}.png")
    best_val = max(hist.history["val_accuracy"])
    print(f"  {key:16} {secs:6.0f}s  epochs={epochs_run:3d} (best {best_epoch})  "
          f"vocab={len(vocab):,}  best val acc={best_val:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="dedup")
    parser.add_argument("--only", help="comma-separated model keys, e.g. lstm_glove50")
    args = parser.parse_args()
    keys = args.only.split(",") if args.only else MODEL_KEYS
    unknown = set(keys) - set(MODEL_KEYS)
    if unknown:
        parser.error(f"unknown model keys: {sorted(unknown)} (choose from {MODEL_KEYS})")

    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    for dataset in datasets:
        train, val = load_split(dataset, "train"), load_split(dataset, "val")
        print(f"\n=== {dataset}: {len(train):,} train / {len(val):,} val ===")
        # Load each embedding once and train every architecture on it.
        for emb in EMBEDDINGS:
            emb_keys = [k for k in keys if k.split("_", 1)[1] == emb]
            if not emb_keys:
                continue
            kv = load_embedding(emb)
            for key in emb_keys:
                train_one(key, dataset, train, val, kv)
    print("\nDone. Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
