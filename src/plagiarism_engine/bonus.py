"""Bonus features for advanced parameter tuning, hybrid hashing, and Persian lemmatization."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Literal

from .simhash import TfidfSimHasher, stable_feature_hash64, make_ngrams


def find_adaptive_lsh_params(num_perm: int, target_threshold: float) -> tuple[int, int]:
    """Find LSH bands (b) and rows (r) that approximate target Jaccard threshold.

    Solves s ~ (1/b)^(1/r) subject to b * r = num_perm.
    """
    if num_perm <= 0:
        raise ValueError("num_perm must be positive.")
    if not 0.0 < target_threshold < 1.0:
        raise ValueError("target_threshold must be between 0 and 1 exclusive.")

    best_b, best_r = 1, num_perm
    min_diff = float("inf")
    
    for b in range(1, num_perm + 1):
        if num_perm % b == 0:
            r = num_perm // b
            thresh = (1.0 / b) ** (1.0 / r)
            diff = abs(thresh - target_threshold)
            if diff < min_diff:
                min_diff = diff
                best_b, best_r = b, r
                
    return best_b, best_r


class HybridTfidfSimHasher(TfidfSimHasher):
    """Extend SimHash with character-level 3-gram fallbacks for typo resilience."""

    def __init__(
        self,
        hash_bits: int = 64,
        ngram_size: int = 1,
        char_ngram_size: int = 3,
        use_char_ngrams: bool = True,
    ) -> None:
        super().__init__(hash_bits=hash_bits, ngram_size=ngram_size)
        self.char_ngram_size = char_ngram_size
        self.use_char_ngrams = use_char_ngrams

    def _extract_features(self, tokens: Sequence[str]) -> list[str]:
        word_features = list(make_ngrams(tokens, self.ngram_size))
        if not self.use_char_ngrams:
            return word_features

        char_features = []
        for token in tokens:
            if len(token) >= self.char_ngram_size:
                for i in range(len(token) - self.char_ngram_size + 1):
                    char_features.append(f"char:{token[i:i + self.char_ngram_size]}")
            else:
                char_features.append(f"char:{token}")
        return word_features + char_features

    def fit(self, token_documents: Iterable[Sequence[str]]) -> HybridTfidfSimHasher:
        documents = [self._extract_features(tokens) for tokens in token_documents]
        self.document_count = len(documents)
        from collections import Counter
        document_frequency: Counter[str] = Counter()
        for features in documents:
            document_frequency.update(set(features))

        self.idf = {
            feature: math.log((self.document_count + 1) / (frequency + 1)) + 1.0
            for feature, frequency in document_frequency.items()
        }
        return self

    def fingerprint(self, tokens: Sequence[str]) -> int:
        features = self._extract_features(tokens)
        if not features:
            return 0
        from collections import Counter
        frequencies = Counter(features)
        accumulator = [0.0] * self.hash_bits

        for feature, frequency in frequencies.items():
            weight = self._weight(feature, frequency)
            feature_hash = stable_feature_hash64(feature)
            for bit in range(self.hash_bits):
                accumulator[bit] += weight if (feature_hash >> bit) & 1 else -weight

        fingerprint = 0
        for bit, value in enumerate(accumulator):
            if value >= 0:
                fingerprint |= 1 << bit
        return fingerprint


def persian_lemmatize(tokens: Iterable[str]) -> tuple[str, ...]:
    """Lightweight rule-based Persian lemmatizer.

    Cleans plural suffixes ('ها', 'ان', 'ات') and verbal tense prefix ('می').
    """
    lemmatized: list[str] = []
    for token in tokens:
        if token == "می":
            continue
            
        word = token
        if word.endswith("ها") and len(word) > 2:
            word = word[:-2]
        elif word.endswith("ان") and len(word) > 2:
            word = word[:-2]
        elif word.endswith("ات") and len(word) > 2:
            word = word[:-2]
            
        word = word.strip("\u200c\u200d\ufeff")
        if word:
            lemmatized.append(word)
    return tuple(lemmatized)
