"""Sparse TF-IDF cosine similarity implemented without external libraries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
import math


SparseVector = tuple[dict[str, float], float]


class TfidfCosineScorer:
    """Create sparse TF-IDF vectors and calculate cosine similarity."""

    def __init__(
        self,
        word_ngram_sizes: tuple[int, ...] = (1,),
        char_ngram_sizes: tuple[int, ...] = (3, 4, 5),
    ) -> None:
        if not word_ngram_sizes and not char_ngram_sizes:
            raise ValueError("At least one feature type is required.")
        if any(size <= 0 for size in word_ngram_sizes):
            raise ValueError("Word n-gram sizes must be positive.")
        if any(size <= 0 for size in char_ngram_sizes):
            raise ValueError("Character n-gram sizes must be positive.")

        self.word_ngram_sizes = word_ngram_sizes
        self.char_ngram_sizes = char_ngram_sizes
        self.document_count = 0
        self.idf: dict[str, float] = {}

    def _extract_features(self, tokens: Sequence[str]) -> tuple[str, ...]:
        clean_tokens = tuple(token for token in tokens if token)
        features: list[str] = []

        for size in self.word_ngram_sizes:
            if len(clean_tokens) < size:
                continue
            for index in range(len(clean_tokens) - size + 1):
                value = "\x1f".join(clean_tokens[index:index + size])
                features.append(f"word-{size}:{value}")

        text = " ".join(clean_tokens)
        for size in self.char_ngram_sizes:
            if len(text) < size:
                continue
            for index in range(len(text) - size + 1):
                features.append(f"char-{size}:{text[index:index + size]}")

        return tuple(features)

    def fit(
        self,
        token_documents: Iterable[Sequence[str]],
    ) -> "TfidfCosineScorer":
        documents = [self._extract_features(tokens) for tokens in token_documents]
        self.document_count = len(documents)
        document_frequency: Counter[str] = Counter()

        for features in documents:
            document_frequency.update(set(features))

        self.idf = {
            feature: math.log((self.document_count + 1) / (frequency + 1)) + 1.0
            for feature, frequency in document_frequency.items()
        }
        return self

    def vector(self, tokens: Sequence[str]) -> SparseVector:
        features = self._extract_features(tokens)
        if not features:
            return {}, 0.0

        frequencies = Counter(features)
        values: dict[str, float] = {}
        unseen_idf = math.log(self.document_count + 1) + 1.0

        for feature, frequency in frequencies.items():
            tf = 1.0 + math.log(frequency)
            idf = self.idf.get(feature, unseen_idf)
            values[feature] = tf * idf

        norm = math.sqrt(sum(value * value for value in values.values()))
        return values, norm

    def transform(
        self,
        token_documents: Iterable[Sequence[str]],
    ) -> list[SparseVector]:
        return [self.vector(tokens) for tokens in token_documents]

    @staticmethod
    def cosine(left: SparseVector, right: SparseVector) -> float:
        left_values, left_norm = left
        right_values, right_norm = right

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        if len(left_values) > len(right_values):
            left_values, right_values = right_values, left_values

        dot_product = sum(
            value * right_values.get(feature, 0.0)
            for feature, value in left_values.items()
        )
        return dot_product / (left_norm * right_norm)
