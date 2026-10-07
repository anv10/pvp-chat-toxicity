"""Raw Kaggle CSVs -> cleaned, weakly labeled, deduplicated train/val/test splits.

Run:  python -m src.prepare_data

Pipeline (each step is there for a reason):
  1. load      LoL Tribunal + Dota 2 chat (Dota read in chunks: the file is 818MB)
  2. clean     drop empty/system lines, normalise text, keep English-ish messages
  3. dedup     collapse identical *normalised* messages into one row with a count.
               Game chat is full of repeats ("gg" alone appears ~50k times). If a
               copy lands in train and another in test, the model is graded on
               text it has memorised and the scores are inflated.
  4. label     weak labels from lexicons + rules (src/labeling.py)
  5. sample    ~33k messages at roughly the paper's 45/30/25 class balance,
               half from each game where possible
  6. split     stratified 80/20 train/test, then 80/20 train/val (fixed seed)
  7. extras    the same pipeline WITHOUT dedup (leakage side experiment), a
               300-message sample for hand-checking, a word2vec corpus, stats.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DOTA_CSV, DOTA_MAX_ROWS, LABELS, LOL_CSV, MANUAL_CHECK_PATH, PROCESSED_DIR,
    SEED, TARGET_BALANCE, TARGET_SIZE, TEST_SIZE, VAL_SIZE, WITH_DUPS_DIR, set_seeds,
)
from src.labeling import weak_label
from src.text_utils import is_english_ish, is_system_line, normalize_text

CHUNK = 1_000_000
MANUAL_CHECK_N = 300


# ---------------------------------------------------------------- 1. load
def iter_raw_chunks():
    """Yield (source, Series of raw messages). Column names were checked in phase 1."""
    lol = pd.read_csv(LOL_CSV, usecols=["message"])["message"]
    yield "lol", lol
    reader = pd.read_csv(DOTA_CSV, usecols=["text"], chunksize=CHUNK, nrows=DOTA_MAX_ROWS)
    for chunk in reader:
        yield "dota", chunk["text"]


# ---------------------------------------------------------------- 2-3. clean + dedup
def clean_and_count(raw: pd.Series, stats: dict) -> pd.DataFrame:
    """Clean one chunk and collapse it to unique normalised texts with counts."""
    stats["rows_raw"] += len(raw)
    raw = raw.dropna().astype(str)
    raw = raw[raw.str.strip() != ""]
    stats["rows_nonempty"] += len(raw)
    raw = raw[~raw.map(is_system_line)]
    stats["rows_not_system"] += len(raw)

    df = pd.DataFrame({"text": raw.str.strip(), "text_norm": raw.map(normalize_text)})
    # Single characters ("k", "?") carry no signal for our task.
    df = df[df["text_norm"].str.len() >= 2]
    df = df[df["text_norm"].map(is_english_ish)]
    stats["rows_english"] += len(df)

    return df.groupby("text_norm").agg(text=("text", "first"), count=("text", "size"))


def build_unique_pool() -> tuple[pd.DataFrame, dict]:
    """Return one row per unique normalised message, with per-source counts."""
    stats = {}
    parts = []
    for source, raw in iter_raw_chunks():
        s = stats.setdefault(source, dict.fromkeys(
            ["rows_raw", "rows_nonempty", "rows_not_system", "rows_english"], 0))
        agg = clean_and_count(raw, s)
        agg["source"] = source
        parts.append(agg.reset_index())
        print(f"  {source}: {s['rows_raw']:,} rows read so far")

    allrows = pd.concat(parts, ignore_index=True)
    # Merge chunk-level counts into per-(text, source) counts...
    per_source = allrows.groupby(["text_norm", "source"]).agg(
        text=("text", "first"), count=("count", "sum")).reset_index()
    for src in stats:
        stats[src]["unique_english"] = int((per_source["source"] == src).sum())

    # ...then into one row per text. A message like "gg" appears in both games;
    # we attribute it to the game where it is more frequent.
    per_source = per_source.sort_values("count", ascending=False)
    pool = per_source.groupby("text_norm").agg(
        text=("text", "first"), source=("source", "first"), count=("count", "sum"))
    return pool.reset_index(), stats


# ---------------------------------------------------------------- 5. sample
def class_targets() -> dict[int, int]:
    total = sum(TARGET_BALANCE.values())
    return {c: round(TARGET_SIZE * w / total) for c, w in TARGET_BALANCE.items()}


def sample_balanced(pool: pd.DataFrame, rng: np.random.Generator,
                    with_dups: bool = False) -> pd.DataFrame:
    """Draw class_targets() messages per class, half from each game if possible.

    with_dups=False: each unique text can be drawn at most once (the clean setup).
    with_dups=True:  every *occurrence* is a candidate, so frequent messages like
                     "gg" get drawn many times, as in a naive random sample.
    """
    picked = []
    for label, target in class_targets().items():
        by_src = {}
        for src in ("lol", "dota"):
            sub = pool[(pool["label"] == label) & (pool["source"] == src)]
            idx = sub.index.to_numpy()
            by_src[src] = np.repeat(idx, sub["count"].to_numpy()) if with_dups else idx

        available = sum(len(v) for v in by_src.values())
        if available < target:
            print(f"  ! class {label}: only {available:,} available (< target {target:,})")
            target = available

        # Half per source; if one source runs short, the other fills the gap.
        n_lol = min(len(by_src["lol"]), target // 2)
        n_dota = min(len(by_src["dota"]), target - n_lol)
        n_lol = min(len(by_src["lol"]), target - n_dota)
        for src, n in (("lol", n_lol), ("dota", n_dota)):
            picked.append(rng.choice(by_src[src], size=n, replace=False))

    rows = pool.loc[np.concatenate(picked)].reset_index(drop=True)
    return rows.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ---------------------------------------------------------------- 6. split
def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stratified so every split keeps the same class balance."""
    trainval, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED)
    train, val = train_test_split(
        trainval, test_size=VAL_SIZE, stratify=trainval["label"], random_state=SEED)
    return {"train": train, "val": val, "test": test}


def save_splits(splits: dict, out_dir) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["text", "text_norm", "label", "rule", "source", "count"]
    info = {}
    for name, df in splits.items():
        df[cols].to_csv(out_dir / f"{name}.csv", index=False)
        info[name] = {"n": len(df),
                      "class_counts": df["label"].value_counts().sort_index().to_dict()}
    # How many test messages also appear (after normalisation) in train?
    train_texts = set(splits["train"]["text_norm"])
    overlap = splits["test"]["text_norm"].isin(train_texts).mean()
    info["test_rows_seen_in_train"] = round(float(overlap), 4)
    return info


# ---------------------------------------------------------------- 7. extras
def export_manual_check(df: pd.DataFrame) -> None:
    """300 random messages for hand-labeling. Never overwrites hand labels.

    If the file exists but nobody has labeled anything yet, it is regenerated so
    its auto_label column reflects the current lexicons.
    """
    if MANUAL_CHECK_PATH.exists():
        filled = pd.read_csv(MANUAL_CHECK_PATH)["manual_label"].notna().sum()
        if filled:
            print(f"  {MANUAL_CHECK_PATH.name} has {filled} hand labels - kept as is "
                  "(delete it to regenerate)")
            return
    sample = df.sample(n=MANUAL_CHECK_N, random_state=SEED)
    out = pd.DataFrame({
        "id": range(1, MANUAL_CHECK_N + 1),
        "source": sample["source"].to_numpy(),
        "text": sample["text"].to_numpy(),
        "auto_label": sample["label"].to_numpy(),
        "manual_label": "",
    })
    out.to_csv(MANUAL_CHECK_PATH, index=False)
    print(f"  wrote {MANUAL_CHECK_PATH} - fill in manual_label with 0/1/2")


def export_w2v_corpus(pool: pd.DataFrame, held_out: set) -> int:
    """Unlabeled chat text for training our own word2vec (phase 4 extension).

    Uses every unique English message EXCEPT val/test ones, so the embeddings
    never see evaluation text.
    """
    corpus = pool.loc[~pool["text_norm"].isin(held_out), "text_norm"]
    path = PROCESSED_DIR / "w2v_corpus.txt"
    path.write_text("\n".join(corpus) + "\n", encoding="utf-8")
    return len(corpus)


def main() -> None:
    set_seeds()
    rng = np.random.default_rng(SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1-3. Loading, cleaning and deduplicating ...")
    pool, source_stats = build_unique_pool()
    print(f"  unique English messages: {len(pool):,} "
          f"(from {int(pool['count'].sum()):,} rows)")

    print("4. Weak labeling ...")
    results = [weak_label(t) for t in pool["text_norm"]]
    pool["label"] = [r.label for r in results]
    pool["rule"] = [r.rule for r in results]
    pool_dist = pool["label"].value_counts().sort_index().to_dict()
    print(f"  label counts in unique pool: {pool_dist}")

    print("5-6. Sampling and splitting (deduplicated) ...")
    dedup = sample_balanced(pool, rng)
    dedup_info = save_splits(split(dedup), PROCESSED_DIR)

    print("7. Side experiment: same pipeline with duplicates left in ...")
    dups = sample_balanced(pool, rng, with_dups=True)
    dups_info = save_splits(split(dups), WITH_DUPS_DIR)
    print(f"  test rows also present in train: dedup={dedup_info['test_rows_seen_in_train']:.1%}"
          f"  with_dups={dups_info['test_rows_seen_in_train']:.1%}")

    export_manual_check(dedup)
    held_out = set(pd.concat([pd.read_csv(PROCESSED_DIR / f"{s}.csv")["text_norm"]
                              for s in ("val", "test")]))
    n_corpus = export_w2v_corpus(pool, held_out)

    stats = {
        "seed": SEED,
        "dota_max_rows": DOTA_MAX_ROWS,
        "per_source": source_stats,
        "unique_english_total": len(pool),
        "rows_english_total": int(pool["count"].sum()),
        "duplicate_rate": round(1 - len(pool) / int(pool["count"].sum()), 4),
        "pool_label_counts": {LABELS[k]: int(v) for k, v in pool_dist.items()},
        "top_rules": pool["rule"].value_counts().head(30).to_dict(),
        "dedup_splits": dedup_info,
        "with_dups_splits": dups_info,
        "with_dups_unique_texts": int(dups["text_norm"].nunique()),
        "w2v_corpus_messages": n_corpus,
    }
    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps({k: stats[k] for k in ("dedup_splits", "with_dups_splits")},
                     indent=2, default=int))
    print("Done. Outputs in data/processed/")


if __name__ == "__main__":
    main()
