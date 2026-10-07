"""How good are the weak labels? Compare them with hand labels.

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

from src.config import LABELS, MANUAL_CHECK_PATH, RESULTS_DIR


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

    y_true, y_auto = df["manual_label"], df["auto_label"]
    acc = accuracy_score(y_true, y_auto)
    kappa = cohen_kappa_score(y_true, y_auto)
    labels = sorted(LABELS)
    names = [LABELS[i] for i in labels]
    cm = confusion_matrix(y_true, y_auto, labels=labels)

    print(f"Hand-labeled rows: {len(df)}")
    print(f"Accuracy (auto vs manual): {acc:.3f}")
    print(f"Cohen's kappa:             {kappa:.3f}\n")
    print("Confusion matrix (rows = manual, cols = auto):")
    print(pd.DataFrame(cm, index=names, columns=names), "\n")
    print(classification_report(y_true, y_auto, labels=labels, target_names=names,
                                zero_division=0))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "n": len(df),
        "accuracy": round(acc, 4),
        "cohen_kappa": round(kappa, 4),
        "confusion_matrix_rows_manual_cols_auto": cm.tolist(),
        "report": classification_report(y_true, y_auto, labels=labels, target_names=names,
                                        zero_division=0, output_dict=True),
    }
    (RESULTS_DIR / "label_agreement.json").write_text(json.dumps(out, indent=2))
    disagreements = df[df["manual_label"] != df["auto_label"]]
    disagreements.to_csv(RESULTS_DIR / "label_disagreements.csv", index=False)
    print(f"Saved results/label_agreement.json and {len(disagreements)} disagreements "
          "to results/label_disagreements.csv")


if __name__ == "__main__":
    main()
