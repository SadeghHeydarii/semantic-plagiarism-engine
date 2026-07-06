"""Banding-based locality-sensitive hashing for MinHash signatures."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Hashable, Iterable, Sequence


class LSHIndex:
    """Store MinHash signatures in band buckets and generate candidate pairs."""

    def __init__(self, num_perm: int = 128, bands: int = 64) -> None:
        if num_perm <= 0 or bands <= 0:
            raise ValueError("num_perm and bands must be positive.")
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands.")
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        self._buckets: list[dict[tuple[int, ...], set[Hashable]]] = [
            defaultdict(set) for _ in range(bands)
        ]
        self._signatures: dict[Hashable, tuple[int, ...]] = {}

    @property
    def approximate_threshold(self) -> float:
        """Return the standard S-curve threshold approximation (1/b)^(1/r)."""

        return (1.0 / self.bands) ** (1.0 / self.rows_per_band)

    def _validate_signature(self, signature: Sequence[int]) -> tuple[int, ...]:
        if len(signature) != self.num_perm:
            raise ValueError(
                f"Expected signature length {self.num_perm}, got {len(signature)}."
            )
        return tuple(signature)

    def band_slices(self, signature: Sequence[int]) -> Iterable[tuple[int, tuple[int, ...]]]:
        checked = self._validate_signature(signature)
        for band in range(self.bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            yield band, checked[start:end]

    def insert(self, document_id: Hashable, signature: Sequence[int]) -> None:
        checked = self._validate_signature(signature)
        if document_id in self._signatures:
            raise ValueError(f"Duplicate document id: {document_id!r}")
        self._signatures[document_id] = checked
        for band, key in self.band_slices(checked):
            self._buckets[band][key].add(document_id)

    def query(self, signature: Sequence[int]) -> set[Hashable]:
        candidates: set[Hashable] = set()
        for band, key in self.band_slices(signature):
            candidates.update(self._buckets[band].get(key, set()))
        return candidates

    def candidate_pairs(self) -> set[tuple[Hashable, Hashable]]:
        pairs: set[tuple[Hashable, Hashable]] = set()
        for bucket_map in self._buckets:
            for document_ids in bucket_map.values():
                if len(document_ids) < 2:
                    continue
                ordered = sorted(document_ids, key=lambda value: str(value))
                pairs.update(combinations(ordered, 2))
        return pairs

    def is_candidate(self, left: Sequence[int], right: Sequence[int]) -> bool:
        left_checked = self._validate_signature(left)
        right_checked = self._validate_signature(right)
        for band in range(self.bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            if left_checked[start:end] == right_checked[start:end]:
                return True
        return False

    @staticmethod
    def candidate_probability(similarity: float, bands: int, rows_per_band: int) -> float:
        if not 0.0 <= similarity <= 1.0:
            raise ValueError("similarity must be between 0 and 1.")
        if bands <= 0 or rows_per_band <= 0:
            raise ValueError("bands and rows_per_band must be positive.")
        return 1.0 - (1.0 - similarity ** rows_per_band) ** bands
