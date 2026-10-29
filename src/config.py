"""Central place for paths, seeds and constants so every script agrees on them."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LEXICON_DIR = DATA_DIR / "lexicons"
PROCESSED_DIR = DATA_DIR / "processed"
WITH_DUPS_DIR = PROCESSED_DIR / "with_dups"  # side experiment: leakage via duplicates
# Train/val/test CSVs that models read. v2 = same messages as DATA_DIR/processed, but with
# the LLM ("gamer") labels; the lexicon labels are kept in a `label_lexicon` column.
SPLITS_DIR = DATA_DIR / "processed_v2"
SPLITS_WITH_DUPS_DIR = SPLITS_DIR / "with_dups"
MANUAL_CHECK_PATH = DATA_DIR / "manual_check.csv"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

LOL_CSV = RAW_DIR / "chatlogs.csv"
DOTA_CSV = RAW_DIR / "dota2_chat_messages.csv"

# The Dota file has 21.6M rows; we only need ~33k messages in the end, so reading
# the first 5M rows (~225k matches) gives more than enough candidates while
# keeping memory and runtime laptop-friendly.
DOTA_MAX_ROWS = 5_000_000

SEED = 42

LABELS = {0: "neutral/positive", 1: "negative", 2: "derogatory"}

# Target dataset size and class balance. The paper had ~33k messages at roughly
# 43% / 30% / 25%; we normalise those to sum to 1.
TARGET_SIZE = 33_000
TARGET_BALANCE = {0: 0.45, 1: 0.30, 2: 0.25}

TEST_SIZE = 0.20  # train+val / test
VAL_SIZE = 0.20   # train / val, taken from the train+val part


def set_seeds(seed: int = SEED) -> None:
    """Fix every random source we use so runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
