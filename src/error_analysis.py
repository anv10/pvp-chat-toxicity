"""Dump misclassified test messages, grouped by (true class, predicted class).

Run:  python -m src.error_analysis                 # best 3-class model by macro F1
      python -m src.error_analysis --model knn_1   # a specific model
      python -m src.error_analysis --all           # every model

Needs `python -m src.evaluate` to have been run first (it writes the predictions).
Output: results/errors/<key>/true<a>_pred<b>.csv, plus a short printout.

Reading these is how we find out *why* a model fails: is it the model, or is
the weak label itself wrong? (e.g. "you feed rene" labeled negative by the
lexicon but arguably just a neutral remark).
"""
import argparse
import json

import pandas as pd

from src.config import RESULTS_DIR

SHOW = 5  # examples printed per class pair


def analyse(key: str, dataset: str = "dedup") -> None:
    path = RESULTS_DIR / "predictions" / dataset / f"{key}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m src.evaluate` first")
    df = pd.read_csv(path, keep_default_na=False)
    meta = json.loads((RESULTS_DIR / "metrics" / dataset / f"{key}.json").read_text())
    names = (["neutral", "toxic"] if meta["task"] == "binary"
             else ["neutral", "negative", "derogatory"])

    errors = df[df["true"] != df["pred"]]
    out_dir = RESULTS_DIR / "errors" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()

    print(f"\n=== {key} ({dataset}): {len(errors):,} / {len(df):,} test messages wrong "
          f"({len(errors) / len(df):.1%}) ===")
    pairs = errors.groupby(["true", "pred"]).size().sort_values(ascending=False)
    for (t, p), n in pairs.items():
        sub = errors[(errors["true"] == t) & (errors["pred"] == p)]
        # Most confident mistakes first, if the model gives probabilities.
        if f"p{p}" in sub:
            sub = sub.sort_values(f"p{p}", ascending=False)
        sub.to_csv(out_dir / f"true{t}_pred{p}.csv", index=False)
        print(f"\n  true={names[t]} -> predicted={names[p]}: {n}")
        for text in sub["text"].head(SHOW):
            print(f"     - {text[:90]}")
    print(f"\n  saved to {out_dir.relative_to(RESULTS_DIR.parent)}/")


def best_model(dataset: str = "dedup") -> str:
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    multi = summary[(summary["dataset"] == dataset) & (summary["task"] == "multiclass")]
    return multi.sort_values("test_macro_f1", ascending=False)["key"].iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump misclassified test examples.")
    parser.add_argument("--model", help="model key (default: best 3-class model)")
    parser.add_argument("--dataset", default="dedup", choices=["dedup", "with_dups"])
    parser.add_argument("--all", action="store_true", help="analyse every model")
    args = parser.parse_args()

    if args.all:
        keys = sorted(p.stem for p in (RESULTS_DIR / "predictions" / args.dataset).glob("*.csv"))
    else:
        keys = [args.model or best_model(args.dataset)]
    for key in keys:
        analyse(key, args.dataset)


if __name__ == "__main__":
    main()
