"""How good are our labels? Compare the lexicon (weak) labels and the final LLM labels with hand labels.

Run:  python -m src.label_agreement

Fill in the `manual_label` column of data/manual_check.csv (0/1/2) first.
Tip: hide the `auto_label` column while labeling so it doesn't bias you.
Rows left blank are skipped, so this also works on a partly labeled file.

Accuracy alone can be misleading when one class dominates. Cohen's kappa
corrects for the agreement you'd get by chance (0 = chance, 1 = perfect;
0.61-0.80 is usually called "substantial" agreement).
"""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix

from src.config import DATA_DIR, LABELS, MANUAL_CHECK_PATH, RESULTS_DIR
from src.text_utils import normalize_text

LLM_LABELS_PATH = DATA_DIR / "llm_labels.csv"


def agreement(df: pd.DataFrame, col: str, tag: str) -> dict:
    """Print + return accuracy / kappa / confusion matrix of label column `col` vs the hand labels."""
    y_true, y_pred = df["manual_label"], df[col]
    labels = sorted(LABELS)
    names = [LABELS[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc, kappa = accuracy_score(y_true, y_pred), cohen_kappa_score(y_true, y_pred)
    print(f"\n=== {tag} labels vs hand labels ({len(df)} rows) ===")
    print(f"Accuracy: {acc:.3f}   Cohen's kappa: {kappa:.3f}\n")
    print("Confusion matrix (rows = manual, cols = this label source):")
    print(pd.DataFrame(cm, index=names, columns=names), "\n")
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))
    return {
        "n": len(df), "accuracy": round(acc, 4), "cohen_kappa": round(kappa, 4),
        "confusion_matrix_rows_manual_cols_label": cm.tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=names,
                                        zero_division=0, output_dict=True),
    }


def main() -> None:
    df = pd.read_csv(MANUAL_CHECK_PATH)
    df = df[df["manual_label"].notna()].copy()
    if df.empty:
        print(f"No manual labels yet - fill in 'manual_label' in {MANUAL_CHECK_PATH}")
        return
    df["manual_label"] = df["manual_label"].astype(int)
    bad = ~df["manual_label"].isin(LABELS)
    if bad.any():
        raise ValueError(f"manual_label must be 0/1/2; bad rows: {df.loc[bad, 'id'].tolist()}")

    # Final dataset labels (LLM, "gamer" labels) for the same messages, matched on normalised text.
    # The original lexicon label is the `auto_label` column.
    llm = pd.read_csv(LLM_LABELS_PATH, keep_default_na=False).drop_duplicates("text_norm")
    df["llm_label"] = df["text"].map(normalize_text).map(dict(zip(llm["text_norm"], llm["label"])))
    n_missing = int(df["llm_label"].isna().sum())
    in_data = df[df["llm_label"].notna()].copy()
    in_data["llm_label"] = in_data["llm_label"].astype(int)
    print(f"{len(in_data)} of {len(df)} hand-labeled rows are in the final dataset ({n_missing} not).")

    out = {"lexicon": agreement(in_data, "auto_label", "Lexicon (weak)"),
           "llm": agreement(in_data, "llm_label", "LLM (final)")}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "label_agreement.json").write_text(json.dumps(out, indent=2))
    disagreements = in_data[in_data["manual_label"] != in_data["llm_label"]]
    disagreements.to_csv(RESULTS_DIR / "label_disagreements.csv", index=False)
    print(f"Saved results/label_agreement.json and {len(disagreements)} LLM-vs-hand "
          "disagreements to results/label_disagreements.csv")


if __name__ == "__main__":
    main()
