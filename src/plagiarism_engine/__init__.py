"""Semantic duplicate and near-plagiarism detection engine."""

from .preprocessing import PreprocessedDocument, TextPreprocessor
from .minhash import MinHasher, exact_jaccard
from .lsh import LSHIndex
from .simhash import TfidfSimHasher, hamming_distance, simhash_similarity
from .ensemble import CalibratedEnsemble, PairFeatureExtractor

__all__ = [
    "PreprocessedDocument",
    "TextPreprocessor",
    "MinHasher",
    "exact_jaccard",
    "LSHIndex",
    "TfidfSimHasher",
    "hamming_distance",
    "simhash_similarity",
    "CalibratedEnsemble",
    "PairFeatureExtractor",
]

__version__ = "1.1.0"
