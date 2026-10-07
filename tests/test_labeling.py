"""Sanity tests for the weak labeler and the English filter.

Run:  python -m pytest tests/

If you edit a lexicon, run these to make sure nothing obvious broke. Add a
case here whenever you find a mislabeled message during error analysis.
"""
import pytest

from src.labeling import weak_label
from src.text_utils import is_english_ish, normalize_text

LABEL_CASES = {
    # neutral / positive, including the paper's context cases
    "ggwp": 0, "thanks": 0, "gg": 0, "nice": 0,
    "my bad": 0, "i suck sorry": 0, "not bad": 0, "too bad": 0,
    # negative attitude
    "gg ez": 1, "GG EZZZZ": 1, "ez!": 1, "noob team": 1, "such a feeder": 1,
    "report this useless guy": 1, "you're bad": 1, "ur so bad": 1, "bad jungle": 1,
    "nooooob": 1, "n00b": 1, "uninstall": 1, "f**k you": 1, "sh*t mid": 1,
    "learn to play!!!": 1, "ur trash!!": 1,
    # derogatory, including obfuscated spellings
    "f*g": 2, "f4g": 2, "f a g": 2, "u r a f a g": 2, "faaaag": 2, "f.a.g": 2,
    "fag!": 2, "n1gg3r": 2, "kys": 2, "noob f*g": 2, "go back to the kitchen": 2,
    "retarded": 2, "stupid bitch": 2, "fake and gay": 2,
    # false-positive traps: slur/insult letters inside innocent words
    "of a game": 0, "half a game": 0, "if a good gank": 0, "f***": 0,
    "class": 0, "assassin": 0, "therapist": 0, "grape": 0, "drape": 0,
    "spicy": 0, "cocoon": 0, "shoe": 0, "shoes": 0, "fuel": 0, "ezreal": 0,
    "homework": 0,
}


@pytest.mark.parametrize("text,expected", LABEL_CASES.items())
def test_weak_label(text, expected):
    result = weak_label(normalize_text(text))
    assert result.label == expected, f"{text!r} -> {result}"


def test_normalize_collapses_repeats_and_case():
    assert normalize_text("  NOOOOOB   team ") == "noob team"
    assert normalize_text("good") == "good"  # real double letters survive


@pytest.mark.parametrize("text,expected", {
    "gg wp": True, "ez mid": True,
    "muss weg": False, "jajaja noob": False, "idi nahui": False,  # other languages
    "?": False, ":)": False,                                        # no letters
    "fkasjfoiashnfiplwqjdf80q23iwkm": False,                        # keyboard mash
    "ладно гг": False,                                              # Cyrillic
}.items())
def test_english_filter(text, expected):
    assert is_english_ish(normalize_text(text)) == expected
