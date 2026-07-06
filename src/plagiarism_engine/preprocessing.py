"""Transparent text preprocessing for English and Persian documents.

The module implements section 2.1 of the project specification using only the
Python standard library.  The steps are deterministic and deliberately kept
small enough to be inspected during grading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Literal, Sequence

Language = Literal["auto", "en", "fa"]
ShortTextPolicy = Literal["single", "empty"]

ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "to", "of", "in", "at", "by", "for", "with", "about",
})

PERSIAN_STOPWORDS: frozenset[str] = frozenset({
    "از", "با", "برای", "به", "تا", "در", "که", "و", "یا", "یک", "را",
})

_CHARACTER_TRANSLATION = str.maketrans({
    "ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا", "ـ": "",
    "\u200c": " ", "\u200d": " ", "\ufeff": " ",
})

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_WHITESPACE_RE = re.compile(r"\s+")
_PERSIAN_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


def _strip_diacritics(text: str) -> str:
    # Remove explicit combining marks (for example Arabic vowel signs) without
    # decomposing precomposed letters such as Persian/Arabic Alef with madda.
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _replace_punctuation_and_symbols(text: str) -> str:
    output: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if char.isspace():
            output.append(" ")
        elif category.startswith(("L", "N")):
            output.append(char)
        else:
            output.append(" ")
    return "".join(output)


def detect_language(text: str) -> Literal["en", "fa"]:
    """Detect the dominant script by counting Persian and Latin letters."""

    persian_count = len(_PERSIAN_LETTER_RE.findall(text))
    latin_count = len(_LATIN_LETTER_RE.findall(text))
    return "fa" if persian_count > latin_count else "en"


@dataclass(frozen=True, slots=True)
class PreprocessedDocument:
    normalized_text: str
    tokens: tuple[str, ...]
    shingles: frozenset[tuple[str, ...]]
    language: Literal["en", "fa"]

    @property
    def is_empty(self) -> bool:
        return not self.tokens


class TextPreprocessor:
    """Normalize, tokenize, remove stop words and create word shingles."""

    def __init__(
        self,
        shingle_size: int = 3,
        language: Language = "auto",
        remove_stopwords: bool = True,
        custom_stopwords: Iterable[str] | None = None,
        short_text_policy: ShortTextPolicy = "single",
        adaptive_short_text: bool = False,
        short_text_token_limit: int = 10,
    ) -> None:
        if not 3 <= shingle_size <= 5:
            raise ValueError("shingle_size must be between 3 and 5.")
        if language not in {"auto", "en", "fa"}:
            raise ValueError("language must be one of: 'auto', 'en', 'fa'.")
        if short_text_policy not in {"single", "empty"}:
            raise ValueError("short_text_policy must be 'single' or 'empty'.")
        if short_text_token_limit <= 0:
            raise ValueError("short_text_token_limit must be positive.")

        self.shingle_size = shingle_size
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.short_text_policy = short_text_policy
        self.adaptive_short_text = adaptive_short_text
        self.short_text_token_limit = short_text_token_limit
        normalized_custom = {
            self.normalize_text(word)
            for word in (custom_stopwords or ())
            if self.normalize_text(word)
        }
        self.custom_stopwords = frozenset(normalized_custom)

    @staticmethod
    def normalize_text(text: str | None) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            raise TypeError("text must be a string or None.")

        text = unicodedata.normalize("NFKC", text)
        text = text.translate(_CHARACTER_TRANSLATION)
        text = text.casefold()
        text = _URL_RE.sub(" ", text)
        text = _EMAIL_RE.sub(" ", text)
        text = _strip_diacritics(text)
        text = _replace_punctuation_and_symbols(text)
        return _WHITESPACE_RE.sub(" ", text).strip()

    def _effective_language(self, normalized_text: str) -> Literal["en", "fa"]:
        return detect_language(normalized_text) if self.language == "auto" else self.language

    def _stopwords_for(self, language: Literal["en", "fa"]) -> frozenset[str]:
        base = ENGLISH_STOPWORDS if language == "en" else PERSIAN_STOPWORDS
        return base | self.custom_stopwords

    def tokenize(
        self,
        text: str | None,
        *,
        already_normalized: bool = False,
    ) -> tuple[str, ...]:
        normalized = text if already_normalized else self.normalize_text(text)
        if not normalized:
            return ()

        language = self._effective_language(normalized)
        tokens = normalized.split()
        keep_stopwords = (
            self.adaptive_short_text
            and len(tokens) <= self.short_text_token_limit
        )
        if self.remove_stopwords and not keep_stopwords:
            stopwords = self._stopwords_for(language)
            tokens = [token for token in tokens if token not in stopwords]
        return tuple(tokens)

    def make_shingles(
        self,
        tokens: Sequence[str],
        *,
        shingle_size: int | None = None,
    ) -> frozenset[tuple[str, ...]]:
        k = self.shingle_size if shingle_size is None else shingle_size
        if not 3 <= k <= 5:
            raise ValueError("shingle_size must be between 3 and 5.")

        clean_tokens = tuple(token for token in tokens if token)
        if not clean_tokens:
            return frozenset()

        if self.adaptive_short_text:
            token_count = len(clean_tokens)
            if token_count <= 6:
                shingle_sizes = (1, 2)
            elif token_count <= 12:
                shingle_sizes = (2, 3)
            else:
                shingle_sizes = (k,)

            shingles: set[tuple[str, ...]] = set()
            for current_size in shingle_sizes:
                if token_count < current_size:
                    shingles.add(clean_tokens)
                    continue
                shingles.update(
                    tuple(clean_tokens[index:index + current_size])
                    for index in range(token_count - current_size + 1)
                )
            return frozenset(shingles)

        if len(clean_tokens) < k:
            if self.short_text_policy == "empty":
                return frozenset()
            return frozenset({clean_tokens})
        return frozenset(
            tuple(clean_tokens[index:index + k])
            for index in range(len(clean_tokens) - k + 1)
        )

    def preprocess(self, text: str | None) -> PreprocessedDocument:
        normalized = self.normalize_text(text)
        language = self._effective_language(normalized)
        tokens = self.tokenize(normalized, already_normalized=True)
        shingles = self.make_shingles(tokens)
        return PreprocessedDocument(normalized, tokens, shingles, language)

    def preprocess_file(self, path: str | Path, *, encoding: str = "utf-8") -> PreprocessedDocument:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        try:
            text = file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            raise ValueError(f"Could not decode {file_path} with encoding {encoding!r}.") from exc
        return self.preprocess(text)

    def preprocess_many(self, documents: Iterable[str | None]) -> list[PreprocessedDocument]:
        return [self.preprocess(document) for document in documents]
