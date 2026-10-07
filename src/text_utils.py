"""Text normalisation and filtering shared by data prep, training and the demo.

Every model sees text that went through `normalize_text`, so the demo must use
the same function on user input or predictions would not match training.
"""
import re
import unicodedata
from functools import lru_cache

from src.config import LEXICON_DIR

_URL_RE = re.compile(r"(https?://|www\.)\S+|\S+\.(com|tv|net|org|ru)\b", re.I)
# 3+ copies of the same character -> 2 copies. "noooob" -> "noob" and
# "ggggg" -> "gg", but real double letters ("good", "gg") stay intact.
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_SPACE_RE = re.compile(r"\s+")
_LETTER_RE = re.compile(r"[a-z]")
_WORD_RE = re.compile(r"[a-z']+")

# Lines the game client writes itself, not something a player typed.
_SYSTEM_RE = re.compile(
    r"has (left|joined|reconnected|disconnected)( the game)?\.?$|^/\w+", re.I
)


def normalize_text(text: str) -> str:
    """Lowercase, unify unicode, drop URLs, collapse char repeats and whitespace.

    Why: game chat writes the same message many ways ("GG", "gg ", "gggg").
    Normalising first means near-identical messages become identical, which is
    what makes deduplication (our leakage fix) actually work.
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _URL_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer used for embeddings / word2vec."""
    return _WORD_RE.findall(text)


@lru_cache(maxsize=1)
def _non_english_markers() -> frozenset[str]:
    path = LEXICON_DIR / "non_english_markers.txt"
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            words.add(line)
    return frozenset(words)


def is_system_line(raw: str) -> bool:
    return bool(_SYSTEM_RE.search(str(raw).strip()))


def is_english_ish(norm: str) -> bool:
    """Cheap English filter (no language-ID model needed).

    1. Mostly ASCII: removes Cyrillic (~1/3 of Dota chat) and other scripts.
    2. Has at least one letter: removes "?", ":)", "+", "100%".
    3. No common Spanish/Portuguese/Russian-translit/German words, from an
       editable marker list, since those languages pass the ASCII check.
    4. No absurdly long tokens: catches keyboard mashing like "fkasjfoiash...".
    """
    if not norm:
        return False
    ascii_ratio = sum(c.isascii() for c in norm) / len(norm)
    if ascii_ratio < 0.9 or not _LETTER_RE.search(norm):
        return False
    tokens = _WORD_RE.findall(norm)
    if any(len(t) > 20 for t in tokens):
        return False
    markers = _non_english_markers()
    return not any(t in markers for t in tokens)
