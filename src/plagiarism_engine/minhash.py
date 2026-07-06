"""MinHash signatures implemented from first principles."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Collection, Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")

# A Mersenne prime large enough for the 64-bit shingle hashes after reduction.
_PRIME = (1 << 61) - 1
_EMPTY_VALUE = _PRIME


def exact_jaccard(left: Collection[T], right: Collection[T]) -> float:
    """Return J(A,B)=|A intersection B|/|A union B|.

    If either set is empty, the similarity is defined as 0.0 to avoid
    false matches between empty or anomalous documents.
    """

    # In this application empty documents are not considered meaningful
    # duplicates; returning zero also avoids empty-empty false positives.
    if not left or not right:
        return 0.0
    union_size = len(set(left) | set(right))
    return len(set(left) & set(right)) / union_size


def _serialize_feature(feature: object) -> bytes:
    if isinstance(feature, tuple):
        value = "\x1f".join(str(part) for part in feature)
    else:
        value = str(feature)
    return value.encode("utf-8", errors="surrogatepass")


def stable_hash64(feature: object) -> int:
    """Produce a deterministic 64-bit integer independent of Python hash seed."""

    digest = hashlib.blake2b(
        _serialize_feature(feature),
        digest_size=8,
        person=b"plag-min",
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


class MinHasher:
    """Create deterministic MinHash signatures using universal hash functions.

    For each permutation i, h_i(x)=(a_i*x+b_i) mod p is evaluated and the
    minimum value over all shingles is retained.
    """

    def __init__(self, num_perm: int = 128, seed: int = 42) -> None:
        if num_perm <= 0:
            raise ValueError("num_perm must be positive.")
        self.num_perm = num_perm
        self.seed = seed
        rng = random.Random(seed)
        self._coefficients = tuple(
            (rng.randrange(1, _PRIME), rng.randrange(0, _PRIME))
            for _ in range(num_perm)
        )

    def signature(self, features: Iterable[object]) -> tuple[int, ...]:
        unique_hashes = {stable_hash64(feature) % _PRIME for feature in features}
        if not unique_hashes:
            return (_EMPTY_VALUE,) * self.num_perm

        signature: list[int] = []
        for a_value, b_value in self._coefficients:
            minimum = min((a_value * value + b_value) % _PRIME for value in unique_hashes)
            signature.append(minimum)
        return tuple(signature)

    @staticmethod
    def similarity(left: Sequence[int], right: Sequence[int]) -> float:
        if len(left) != len(right):
            raise ValueError("MinHash signatures must have equal lengths.")
        if not left:
            raise ValueError("MinHash signatures must not be empty.")
        equal = sum(a == b for a, b in zip(left, right))
        return equal / len(left)
