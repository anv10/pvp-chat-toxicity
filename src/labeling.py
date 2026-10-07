"""Weak labeling: assign 0/1/2 to chat messages with lexicons + regex rules.

Our datasets have no message-level labels, and hand-labeling 33k messages (as
the paper did) isn't possible in a week. So we label automatically with rules,
then measure how good the rules are on a 300-message hand-checked sample
(see src/label_agreement.py).

Decision order (first match wins):
  1. remove "neutral override" phrases ("my bad", "i suck", "not bad", ...)
  2. derogatory lexicon matches        -> 2
  3. negative lexicon or context rule  -> 1
  4. otherwise                         -> 0
Derogatory is checked first because a message like "noob f*g" is both
negative and derogatory, and the more severe class should win.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.config import LEXICON_DIR

# Common leetspeak substitutions players use to dodge chat filters.
LEET = {
    "a": "4@",
    "e": "3",
    "i": "1!|",
    "o": "0",
    "s": "5$",
    "t": "7+",
    "g": "9",
    "b": "8",
}
_LETTER = r"a-z0-9@$!|+*"  # chars that can be part of an (obfuscated) word


def load_lexicon(name: str) -> list[str]:
    """Read a lexicon file: one term per line, '#' comments, blank lines ignored."""
    path = Path(LEXICON_DIR) / name
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            terms.append(line)
    return terms


def _term_to_regex(term: str) -> tuple[str, str]:
    """Turn a lexicon term into (body, right_boundary) regex parts that tolerate obfuscation.

    "fag" becomes roughly  f+ [a4@*]+ [g9*]+
      - each letter may be swapped for its leetspeak digit/symbol
      - each letter may repeat ("faaag")
      - letters after the first may be masked with '*' ("f*g")
    A trailing '*' in the lexicon means "prefix match" (no right boundary).
    """
    prefix = term.endswith("*")
    term = term.rstrip("*")
    parts = []
    for i, ch in enumerate(term):
        if ch == " ":
            parts.append(r"\s+")
            continue
        options = re.escape(ch + LEET.get(ch, ""))
        if i > 0:
            options += r"\*"
        parts.append(f"[{options}]+")
    body = "".join(parts)
    # Only a following LETTER blocks the match, so "trash!!" and "noob123" still match.
    right = "" if prefix else "(?![a-z])"
    return body, right


class Lexicon:
    """A list of terms compiled into one big regex, so matching stays fast."""

    def __init__(self, terms: list[str], boundaries: bool = True):
        self.terms = terms
        pieces = []
        for i, term in enumerate(terms):
            body, right = _term_to_regex(term)
            if boundaries:
                # Lookarounds instead of \b so "noob123" still matches "noob".
                pieces.append(f"(?P<t{i}>(?<![a-z]){body}{right})")
            else:
                pieces.append(f"(?P<t{i}>{body})")
        self.regex = re.compile("|".join(pieces))

    @staticmethod
    def _too_masked(span: str) -> bool:
        # "f**k" is fine, but "f***" could be anything, so require at least
        # two real characters in a masked match.
        return span.count("*") > len(span.replace(" ", "")) - 2

    def finditer(self, text: str):
        for m in self.regex.finditer(text):
            if not self._too_masked(m.group()):
                yield self.terms[int(m.lastgroup[1:])]

    def search(self, text: str) -> str | None:
        """Return the first matching lexicon term, or None."""
        return next(self.finditer(text), None)

    def findall(self, text: str) -> list[str]:
        return list(self.finditer(text))


# "bad" is only negative when aimed at someone: "you're bad", "ur so bad",
# "bad jungle". "my bad" was already removed by the overrides.
_BAD_AT_SOMEONE = re.compile(
    r"\b(you|u|ur|youre|ure|yall|team|guys|he|hes|she|shes|they|theyre)\s+"
    r"((are|r|is|so|really|very|fucking|fkin|too)\s+)*bad\b"
    r"|\bbad\s+(team|player|players|jungle|jungler|jg|mid|top|adc|bot|supp|"
    r"support|carry|ganks?|plays?|teammates?)\b"
)
CONTEXT_RULES = {"bad": _BAD_AT_SOMEONE}

# Runs of 3+ single characters separated by spaces/dots, e.g. "f a g", "n.i.g".
_SPACED_RUN = re.compile(rf"(?<![^\s.])(?:[{_LETTER}][\s.]+){{2,}}[{_LETTER}](?![^\s.])")
# Separators *inside* a word: "f.a.g", "n-i-g-g-a", "f_a_g".
_INNER_SEP = re.compile(rf"(?<=[{_LETTER}])[._\-]+(?=[{_LETTER}])")


@dataclass(frozen=True)
class LabelResult:
    label: int
    rule: str  # why this label was given, e.g. "derog:fag", "neg:noob*", "none"


class WeakLabeler:
    def __init__(self):
        self.derog = Lexicon(load_lexicon("derogatory.txt"))
        # Same derogatory terms but without word boundaries, used only on the
        # joined-up spaced-letter runs ("u r a f a g" -> "urafag").
        self.derog_substring = Lexicon(load_lexicon("derogatory.txt"), boundaries=False)
        self.neg = Lexicon(load_lexicon("negative.txt"))
        self.overrides = Lexicon(load_lexicon("neutral_overrides.txt"))

    def _remove_overrides(self, text: str) -> str:
        return self.overrides.regex.sub(" ; ", text)

    def label(self, norm_text: str) -> LabelResult:
        """Label one *normalised* message (see src.text_utils.normalize_text)."""
        # Apostrophes vary ("you're"/"youre"), so the lexicons are written
        # without them and we strip them here.
        text = norm_text.replace("'", "").replace("’", "")
        text = self._remove_overrides(text)

        # --- class 2: derogatory ---
        term = self.derog.search(text) or self.derog.search(_INNER_SEP.sub("", text))
        if term:
            return LabelResult(2, f"derog:{term}")
        for run in _SPACED_RUN.finditer(text):
            joined = re.sub(r"[\s.]+", "", run.group())
            term = self.derog_substring.search(joined)
            if term:
                return LabelResult(2, f"derog_spaced:{term}")

        # --- class 1: negative attitude ---
        term = self.neg.search(text)
        if term:
            return LabelResult(1, f"neg:{term}")
        for name, rule in CONTEXT_RULES.items():
            if rule.search(text):
                return LabelResult(1, f"neg_ctx:{name}")

        return LabelResult(0, "none")


@lru_cache(maxsize=1)
def get_labeler() -> WeakLabeler:
    return WeakLabeler()


def weak_label(norm_text: str) -> LabelResult:
    return get_labeler().label(norm_text)
