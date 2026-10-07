"""Keras text classifiers: feedforward NN and LSTM on top of word embeddings.

Text goes in as a sequence of word ids:
    "gg ez noob" -> tokenize -> ["gg", "ez", "noob"] -> [5, 812, 40, 0, 0, ...]
The Embedding layer maps each id to its GloVe / word2vec vector. We FREEZE the
embeddings (trainable=False), so the comparison between GloVe and our own
word2vec measures what each embedding already knows about game chat, rather
than what the network could re-learn from 21k messages.
"""
import os

import numpy as np

from src.config import MODELS_DIR
from src.text_utils import tokenize

# Hide TensorFlow's C++ info/warning logs; real errors still show.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PAD, OOV = 0, 1   # id 0 = padding, id 1 = word not in the embedding
MAX_LEN = 30      # ~99% of messages are shorter than this; longer ones are cut


def build_vocab(texts, kv) -> dict[str, int]:
    """Word -> id for every training word the embedding knows; anything else is OOV."""
    vocab = {}
    for text in texts:
        for tok in tokenize(text):
            if tok not in vocab and tok in kv.key_to_index:
                vocab[tok] = len(vocab) + 2
    return vocab


def encode(texts, vocab: dict[str, int], max_len: int = MAX_LEN) -> np.ndarray:
    X = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, text in enumerate(texts):
        ids = [vocab.get(tok, OOV) for tok in tokenize(text)][:max_len]
        X[i, :len(ids)] = ids
    return X


def embedding_matrix(vocab: dict[str, int], kv) -> np.ndarray:
    """Row i = vector of the word with id i. PAD and OOV rows stay all-zero."""
    matrix = np.zeros((len(vocab) + 2, kv.vector_size), dtype=np.float32)
    for word, idx in vocab.items():
        matrix[idx] = kv[word]
    return matrix


def build_model(arch: str, emb_matrix: np.ndarray):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(MAX_LEN,), dtype="int32")
    x = layers.Embedding(
        input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1],
        embeddings_initializer=keras.initializers.Constant(emb_matrix),
        trainable=False, mask_zero=True,  # mask: padding is ignored downstream
    )(inputs)
    if arch == "ff":
        # Paper's feedforward net: ReLU -> ReLU -> softmax. A feedforward layer
        # needs a fixed-size input, so we average the word vectors first.
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation="relu")(x)
    elif arch == "lstm":
        # The LSTM reads words in order, so "my bad" and "bad mid" can differ.
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.3)(x)
    else:
        raise ValueError(arch)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class KerasTextClassifier:
    """sklearn-style wrapper (.predict / .predict_proba on raw text).

    The Keras model itself lives in its own .keras file and is loaded lazily, so
    the joblib bundle stays small and TensorFlow is only imported when needed.
    """

    def __init__(self, vocab: dict[str, int], model_file: str, max_len: int = MAX_LEN):
        self.vocab = vocab
        self.model_file = model_file  # relative to models/
        self.max_len = max_len
        self._keras = None

    def _model(self):
        if self._keras is None:
            from tensorflow import keras
            self._keras = keras.models.load_model(MODELS_DIR / self.model_file)
        return self._keras

    def predict_proba(self, texts) -> np.ndarray:
        X = encode(list(texts), self.vocab, self.max_len)
        # Calling the model directly (instead of .predict) is much faster for the
        # one-message-at-a-time demo and avoids TensorFlow re-tracing warnings.
        model = self._model()
        return np.concatenate([model(X[i:i + 1024], training=False).numpy()
                               for i in range(0, max(len(X), 1), 1024)])

    def predict(self, texts) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_keras"] = None  # never pickle the loaded TF model
        return state
