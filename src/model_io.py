"""Save/load trained models in one format, so evaluate.py and the demo can
treat every model the same way regardless of library.

A saved model is a "bundle": the fitted model plus metadata (name, task, ...).
"""
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from src.config import MODELS_DIR

DATASETS = ("dedup", "with_dups")


@dataclass
class ModelBundle:
    key: str            # short id, e.g. "svm_rbf_bin"; matches results/paper_results.csv
    name: str           # human-readable, e.g. "SVM (RBF kernel)"
    family: str         # "SVM", "Naive Bayes", "KNN", ...
    features: str       # "word", "char", "word+char", "lexicon", "glove50", ...
    task: str           # "binary" (0 vs 1+2) or "multiclass" (0/1/2)
    model: object       # anything with .predict(list_of_texts)
    train_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def class_names(self) -> list[str]:
        if self.task == "binary":
            return ["neutral/positive", "toxic (1+2)"]
        return ["neutral/positive", "negative", "derogatory"]

    def predict(self, texts) -> np.ndarray:
        return np.asarray(self.model.predict(list(texts)))

    def predict_proba(self, texts) -> np.ndarray | None:
        """Class probabilities, or None for models that don't give them (SVM)."""
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(list(texts)))
        return None


def model_dir(dataset: str) -> Path:
    return MODELS_DIR / dataset


def save_bundle(bundle: ModelBundle, dataset: str) -> Path:
    path = model_dir(dataset) / f"{bundle.key}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: Path) -> ModelBundle:
    return joblib.load(path)


def list_bundles(dataset: str) -> list[Path]:
    return sorted(model_dir(dataset).glob("*.joblib"))


def binarize(y) -> np.ndarray:
    """0 stays 0; negative (1) and derogatory (2) both become 1 = toxic."""
    return (np.asarray(y) > 0).astype(int)
