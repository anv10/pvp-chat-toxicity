"""Score every saved model on the test set and build the summary tables.

Run:  python -m src.evaluate

For every model in models/<dataset>/ this writes:
  results/metrics/<dataset>/<key>.json      accuracy, macro F1, per-class P/R/F1, confusion matrix
  results/predictions/<dataset>/<key>.csv   test predictions (used by error_analysis.py)
  results/figures/cm_<dataset>_<key>.png    confusion matrix plot
and across all models:
  results/summary.csv / summary.md          comparison with each other and with the paper
  results/figures/leakage_effect.png        dedup vs with-duplicates scores

Why macro F1 and not only accuracy: accuracy is dominated by the biggest class
(neutral, 45%). Macro F1 averages the F1 of each class equally, so a model that
is bad at spotting derogatory messages can't hide behind good neutral scores.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.config import FIGURES_DIR, RESULTS_DIR
from src.model_io import DATASETS, ModelBundle, binarize, list_bundles, load_bundle
from src.plot_style import SEQUENTIAL_CMAP, apply_style
from src.train_ml import load_split

# Order of rows in the summary table (paper order, then our additions).
FAMILY_ORDER = ["SVM", "Naive Bayes", "KNN", "Random Forest", "Logistic Regression",
                "Feedforward NN", "LSTM", "Transformer"]


def score(y_true, y_pred, n_classes: int) -> dict:
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "per_class": {str(i): {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(cm: np.ndarray, class_names: list[str], title: str, path) -> None:
    """Colour = share of the TRUE class (row-normalised); text = count + share."""
    cm = np.asarray(cm)
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.6 * n + 1.6, 1.4 * n + 1.2))
    ax.imshow(share, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color="white" if share[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(n), class_names, rotation=20, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (weak label)")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def evaluate_bundle(bundle: ModelBundle, dataset: str, test: pd.DataFrame,
                    val: pd.DataFrame, train_texts: set) -> dict:
    binary = bundle.task == "binary"
    n_classes = 2 if binary else 3
    results = {}
    for split, df in (("val", val), ("test", test)):
        y_true = binarize(df["label"]) if binary else df["label"].to_numpy()
        y_pred = bundle.predict(df["text_norm"])
        results[split] = score(y_true, y_pred, n_classes)
        if split == "test":
            # Leakage check: accuracy on test messages whose exact (normalised)
            # text also occurs in train vs. genuinely new messages.
            seen = df["text_norm"].isin(train_texts).to_numpy()
            results["test"]["share_seen_in_train"] = seen.mean()
            results["test"]["accuracy_seen"] = (
                accuracy_score(y_true[seen], y_pred[seen]) if seen.any() else None)
            results["test"]["accuracy_unseen"] = accuracy_score(y_true[~seen], y_pred[~seen])
            preds = df[["text", "text_norm", "source", "label"]].copy()
            preds["true"] = y_true
            preds["pred"] = y_pred
            proba = bundle.predict_proba(df["text_norm"])
            if proba is not None:
                for c in range(proba.shape[1]):
                    preds[f"p{c}"] = proba[:, c].round(4)
            out = RESULTS_DIR / "predictions" / dataset / f"{bundle.key}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            preds.to_csv(out, index=False)

    plot_confusion(results["test"]["confusion_matrix"], bundle.class_names,
                   f"{bundle.name}\n({dataset}, test set)",
                   FIGURES_DIR / f"cm_{dataset}_{bundle.key}.png")
    meta = {"key": bundle.key, "name": bundle.name, "family": bundle.family,
            "features": bundle.features, "task": bundle.task, "dataset": dataset,
            "train_seconds": bundle.train_seconds, **bundle.extra}
    out = RESULTS_DIR / "metrics" / dataset / f"{bundle.key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**meta, **results}, indent=2, default=float))
    return {**meta, **results}


def summary_row(r: dict) -> dict:
    t = r["test"]
    pc = t["per_class"]
    return {
        "dataset": r["dataset"], "key": r["key"], "model": r["name"], "family": r["family"],
        "task": r["task"], "features": r["features"],
        "test_accuracy": t["accuracy"], "test_macro_f1": t["macro_f1"],
        "f1_class0": pc["0"]["f1"], "f1_class1": pc["1"]["f1"],
        "f1_class2": pc["2"]["f1"] if "2" in pc else np.nan,
        "val_accuracy": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"],
        "test_acc_seen": t["accuracy_seen"], "test_acc_unseen": t["accuracy_unseen"],
        "train_seconds": r["train_seconds"],
    }


def fmt(x, digits=3) -> str:
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def write_markdown(summary: pd.DataFrame) -> str:
    main = summary[summary["dataset"] == "dedup"]
    lines = [
        "# Results summary",
        "",
        "All scores are on the held-out **test set** (6,600 messages). Labels are our weak "
        "labels (see `src/labeling.py`), so scores measure agreement with those rules, not "
        "with human judgement. Binary models predict neutral (0) vs toxic (1+2), so their "
        "class-1 F1 column is the *toxic* class and class 2 is blank.",
        "",
        "## Main results (deduplicated data) vs the paper",
        "",
        "| Model | Task | Features | Accuracy | Macro F1 | F1 c0 | F1 c1 | F1 c2 "
        "| Paper acc. | Paper F1 c0/c1/c2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in main.iterrows():
        paper_f1 = "/".join(fmt(r[c], 2) for c in ("paper_f1_0", "paper_f1_1", "paper_f1_2")
                            if not pd.isna(r[c]))
        lines.append(
            f"| {r['model']} | {r['task']} | {r['features']} | {fmt(r['test_accuracy'])} "
            f"| {fmt(r['test_macro_f1'])} | {fmt(r['f1_class0'])} | {fmt(r['f1_class1'])} "
            f"| {fmt(r['f1_class2'])} | {fmt(r['paper_accuracy'], 4)} | {paper_f1} |")

    both = summary.pivot_table(index=["key", "model"], columns="dataset",
                               values=["test_accuracy", "test_macro_f1", "test_acc_seen",
                                       "test_acc_unseen"], sort=False)
    if "with_dups" in summary["dataset"].values:
        lines += [
            "",
            "## Side experiment: what duplicates do to the scores",
            "",
            "Same models and the same 33k size/balance, but sampled without deduplication, "
            "so about 30% of test messages also appear word-for-word in train. The last two "
            "columns split the with-dups test set into those *seen* messages and the "
            "*unseen* rest: near-perfect on seen messages is memorisation, not skill.",
            "",
            "| Model | Acc. dedup | Acc. with dups | Δ acc. | Macro F1 dedup "
            "| Macro F1 with dups | Δ macro F1 | With dups: acc. seen | With dups: acc. unseen |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for (key, model), r in both.iterrows():
            a0, a1 = r[("test_accuracy", "dedup")], r[("test_accuracy", "with_dups")]
            f0, f1 = r[("test_macro_f1", "dedup")], r[("test_macro_f1", "with_dups")]
            seen, unseen = r[("test_acc_seen", "with_dups")], r[("test_acc_unseen", "with_dups")]
            lines.append(f"| {model} | {fmt(a0)} | {fmt(a1)} | {a1 - a0:+.3f} "
                         f"| {fmt(f0)} | {fmt(f1)} | {f1 - f0:+.3f} | {fmt(seen)} | {fmt(unseen)} |")
    return "\n".join(lines) + "\n"


def plot_leakage(summary: pd.DataFrame) -> None:
    """Dumbbell chart: macro F1 per model, dedup vs with duplicates."""
    wide = summary.pivot_table(index="model", columns="dataset", values="test_macro_f1",
                               sort=False)
    if "with_dups" not in wide:
        return
    wide = wide.dropna().sort_values("dedup")
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(wide) + 1.2))
    y = np.arange(len(wide))
    ax.hlines(y, wide["dedup"], wide["with_dups"], color="#bbbbbb", linewidth=2)
    ax.scatter(wide["dedup"], y, s=55, color="#2a78d6", zorder=3,
               edgecolor="white", linewidth=1.5, label="deduplicated (honest)")
    ax.scatter(wide["with_dups"], y, s=55, color="#eb6834", zorder=3,
               edgecolor="white", linewidth=1.5, label="duplicates kept (leaky)")
    ax.set_yticks(y, wide.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test macro F1")
    ax.set_title("Duplicates in both train and test inflate every model's score")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES_DIR / "leakage_effect.png")
    plt.close(fig)


def plot_paper_comparison(summary: pd.DataFrame) -> None:
    """Our test accuracy (deduplicated) vs the accuracy reported in the paper."""
    df = summary[(summary["dataset"] == "dedup") & summary["paper_accuracy"].notna()]
    df = df.sort_values("paper_accuracy")
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(df) + 1.2))
    y = np.arange(len(df))
    ax.hlines(y, df["test_accuracy"], df["paper_accuracy"], color="#bbbbbb", linewidth=2)
    ax.scatter(df["paper_accuracy"], y, s=55, color="#8a8a8a", zorder=3,
               edgecolor="white", linewidth=1.5, label="paper (reported)")
    ax.scatter(df["test_accuracy"], y, s=55, color="#2a78d6", zorder=3,
               edgecolor="white", linewidth=1.5, label="ours (deduplicated test set)")
    ax.set_yticks(y, df["model"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test accuracy")
    ax.set_title("Our replication vs the paper's reported accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.savefig(FIGURES_DIR / "paper_comparison.png")
    plt.close(fig)


def main() -> None:
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        paths = list_bundles(dataset)
        if not paths:
            print(f"(no models in models/{dataset}/ - skipping)")
            continue
        test, val = load_split(dataset, "test"), load_split(dataset, "val")
        train_texts = set(load_split(dataset, "train")["text_norm"])
        print(f"\n=== {dataset}: {len(paths)} models, {len(test):,} test messages ===")
        for path in paths:
            r = evaluate_bundle(load_bundle(path), dataset, test, val, train_texts)
            rows.append(summary_row(r))
            print(f"  {r['key']:22} acc={r['test']['accuracy']:.3f}  "
                  f"macroF1={r['test']['macro_f1']:.3f}")

    summary = pd.DataFrame(rows)
    paper = pd.read_csv(RESULTS_DIR / "paper_results.csv").dropna(subset=["our_key"])
    paper = paper.rename(columns={"our_key": "key", "accuracy": "paper_accuracy",
                                  "f1_class0": "paper_f1_0", "f1_class1": "paper_f1_1",
                                  "f1_class2": "paper_f1_2"})
    summary = summary.merge(paper[["key", "paper_accuracy", "paper_f1_0", "paper_f1_1",
                                   "paper_f1_2"]], on="key", how="left")
    summary["_fam"] = summary["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    summary = (summary.sort_values(["dataset", "_fam", "key"]).drop(columns="_fam")
               .reset_index(drop=True))
    summary.round(4).to_csv(RESULTS_DIR / "summary.csv", index=False)
    (RESULTS_DIR / "summary.md").write_text(write_markdown(summary))
    plot_leakage(summary)
    plot_paper_comparison(summary)
    print("\nWrote results/summary.csv, results/summary.md and figures in results/figures/")


if __name__ == "__main__":
    main()
