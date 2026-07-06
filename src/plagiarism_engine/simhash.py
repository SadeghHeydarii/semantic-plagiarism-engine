"""TF-IDF weighted 64-bit SimHash implemented without external libraries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
import hashlib
import math


def stable_feature_hash64(feature: str) -> int:
    digest = hashlib.blake2b(
        feature.encode("utf-8", errors="surrogatepass"),
        digest_size=8,
        person=b"plag-sim",
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def make_ngrams(tokens: Sequence[str], ngram_size: int = 1) -> tuple[str, ...]:
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive.")
    if not tokens:
        return ()
    if len(tokens) < ngram_size:
        return ("\x1f".join(tokens),)
    return tuple(
        "\x1f".join(tokens[index:index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    )


class TfidfSimHasher:
    """Fit corpus IDF values and create weighted 64-bit SimHash fingerprints."""

    def __init__(self, hash_bits: int = 64, ngram_size: int = 1) -> None:
        if hash_bits != 64:
            raise ValueError("This project implementation uses exactly 64 bits.")
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive.")
        self.hash_bits = hash_bits
        self.ngram_size = ngram_size
        self.document_count = 0
        self.idf: dict[str, float] = {}

    def fit(self, token_documents: Iterable[Sequence[str]]) -> "TfidfSimHasher":
        documents = [make_ngrams(tuple(tokens), self.ngram_size) for tokens in token_documents]
        self.document_count = len(documents)
        document_frequency: Counter[str] = Counter()
        for features in documents:
            document_frequency.update(set(features))

        # Smoothed IDF avoids division by zero and keeps all weights positive.
        self.idf = {
            feature: math.log((self.document_count + 1) / (frequency + 1)) + 1.0
            for feature, frequency in document_frequency.items()
        }
        return self

    def _weight(self, feature: str, frequency: int) -> float:
        # Sub-linear TF follows 1+log(tf), multiplied by smoothed IDF.
        tf = 1.0 + math.log(frequency)
        idf = self.idf.get(feature)
        if idf is None:
            idf = math.log((self.document_count + 1) / 1.0) + 1.0
        return tf * idf

    def fingerprint(self, tokens: Sequence[str]) -> int:
        features = make_ngrams(tuple(tokens), self.ngram_size)
        if not features:
            return 0
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

    def fingerprints(self, token_documents: Iterable[Sequence[str]]) -> list[int]:
        return [self.fingerprint(tokens) for tokens in token_documents]


def hamming_distance(left: int, right: int) -> int:
    if left < 0 or right < 0:
        raise ValueError("SimHash values must be non-negative integers.")
    return (left ^ right).bit_count()


def simhash_similarity(left: int, right: int, hash_bits: int = 64) -> float:
    if hash_bits <= 0:
        raise ValueError("hash_bits must be positive.")
    return 1.0 - hamming_distance(left, right) / hash_bits
