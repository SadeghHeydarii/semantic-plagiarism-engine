"""Evaluation utilities for the project pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Iterable

from .bonus import HybridTfidfSimHasher
from .dataset import LabeledPair
from .lsh import LSHIndex
from .minhash import MinHasher, exact_jaccard
from .preprocessing import TextPreprocessor
from .simhash import TfidfSimHasher, hamming_distance, simhash_similarity
from .tfidf_cosine import TfidfCosineScorer


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def classification_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
) -> ClassificationMetrics:
    y_true = list(labels)
    y_pred = list(predictions)
    if len(y_true) != len(y_pred):
        raise ValueError("labels and predictions must have equal lengths.")
    if not y_true:
        raise ValueError("At least one labeled example is required.")

    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true)
    return ClassificationMetrics(precision, recall, f1, accuracy, tp, fp, tn, fn)


def evaluate_pairs(
    pairs: list[LabeledPair],
    *,
    preprocessor: TextPreprocessor,
    minhasher: MinHasher,
    bands: int,
    jaccard_threshold: float,
    simhash_max_distance: int,
    simhash_ngram_size: int = 1,
    simhash_mode: str = "standard",
    cosine_threshold: float | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate LSH, SimHash, and optional TF-IDF cosine."""

    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1.")
    if not 0 <= simhash_max_distance <= 64:
        raise ValueError("simhash_max_distance must be between 0 and 64.")
    if simhash_mode not in {"standard", "hybrid"}:
        raise ValueError("simhash_mode must be either 'standard' or 'hybrid'.")
    if cosine_threshold is not None and not 0.0 <= cosine_threshold <= 1.0:
        raise ValueError("cosine_threshold must be between 0 and 1.")

    processed_a = [preprocessor.preprocess(pair.text_a) for pair in pairs]
    processed_b = [preprocessor.preprocess(pair.text_b) for pair in pairs]
    labels = [pair.label for pair in pairs]

    # MinHash + LSH + exact Jaccard
    lsh = LSHIndex(num_perm=minhasher.num_perm, bands=bands)
    lsh_predictions: list[int] = []
    lsh_details: list[dict[str, object]] = []
    lsh_candidate_count = 0
    positive_pairs = sum(labels)
    candidate_positive_pairs = 0

    start = perf_counter()
    for pair, left, right in zip(pairs, processed_a, processed_b):
        signature_a = minhasher.signature(left.shingles)
        signature_b = minhasher.signature(right.shingles)
        candidate = bool(left.shingles and right.shingles) and lsh.is_candidate(
            signature_a,
            signature_b,
        )
        lsh_candidate_count += int(candidate)
        if candidate and pair.label == 1:
            candidate_positive_pairs += 1

        exact = exact_jaccard(left.shingles, right.shingles)
        estimate = minhasher.similarity(signature_a, signature_b)
        prediction = int(candidate and exact >= jaccard_threshold)
        lsh_predictions.append(prediction)
        lsh_details.append({
            "pair_id": pair.pair_id,
            "label": pair.label,
            "lsh_candidate": int(candidate),
            "jaccard": round(exact, 6),
            "minhash_similarity": round(estimate, 6),
            "lsh_prediction": prediction,
        })

    lsh_runtime = perf_counter() - start
    lsh_metrics = classification_metrics(labels, lsh_predictions)
    candidate_recall = (
        candidate_positive_pairs / positive_pairs
        if positive_pairs
        else 0.0
    )

    # Standard or Hybrid SimHash
    if simhash_mode == "hybrid":
        simhasher = HybridTfidfSimHasher(
            ngram_size=simhash_ngram_size,
            char_ngram_size=3,
            use_char_ngrams=True,
        )
        simhash_method_name = "Hybrid TF-IDF SimHash"
    else:
        simhasher = TfidfSimHasher(ngram_size=simhash_ngram_size)
        simhash_method_name = "TF-IDF weighted SimHash"

    all_token_documents = (
        [doc.tokens for doc in processed_a]
        + [doc.tokens for doc in processed_b]
    )
    simhasher.fit(all_token_documents)
    sim_predictions: list[int] = []

    start = perf_counter()
    for index, (pair, left, right) in enumerate(
        zip(pairs, processed_a, processed_b)
    ):
        fingerprint_a = simhasher.fingerprint(left.tokens)
        fingerprint_b = simhasher.fingerprint(right.tokens)
        distance = hamming_distance(fingerprint_a, fingerprint_b)
        similarity = simhash_similarity(fingerprint_a, fingerprint_b)
        prediction = int(
            bool(left.tokens and right.tokens)
            and distance <= simhash_max_distance
        )
        sim_predictions.append(prediction)
        lsh_details[index].update({
            "simhash_hamming": distance,
            "simhash_similarity": round(similarity, 6),
            "simhash_prediction": prediction,
            "error_lsh": int(lsh_predictions[index] != pair.label),
            "error_simhash": int(prediction != pair.label),
            "text_a": pair.text_a,
            "text_b": pair.text_b,
        })

    sim_runtime = perf_counter() - start
    sim_metrics = classification_metrics(labels, sim_predictions)
    total = len(pairs)

    metric_rows: list[dict[str, object]] = [
        {
            "method": "MinHash+LSH+exact-Jaccard",
            **{
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in lsh_metrics.as_dict().items()
            },
            "runtime_seconds": round(lsh_runtime, 6),
            "pairs_evaluated": total,
            "candidate_pairs": lsh_candidate_count,
            "candidate_recall": round(candidate_recall, 6),
            "comparison_reduction": round(
                1.0 - lsh_candidate_count / total,
                6,
            ),
            "jaccard_threshold": jaccard_threshold,
            "simhash_max_distance": "",
            "cosine_threshold": "",
            "num_perm": minhasher.num_perm,
            "bands": bands,
        },
        {
            "method": simhash_method_name,
            **{
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in sim_metrics.as_dict().items()
            },
            "runtime_seconds": round(sim_runtime, 6),
            "pairs_evaluated": total,
            "candidate_pairs": total,
            "candidate_recall": "",
            "comparison_reduction": 0.0,
            "jaccard_threshold": "",
            "simhash_max_distance": simhash_max_distance,
            "cosine_threshold": "",
            "num_perm": "",
            "bands": "",
        },
    ]

    # Optional TF-IDF Cosine bonus
    if cosine_threshold is not None:
        cosine_start = perf_counter()
        cosine_scorer = TfidfCosineScorer(
            word_ngram_sizes=(1,),
            char_ngram_sizes=(3, 4, 5),
        )
        cosine_scorer.fit(all_token_documents)
        cosine_vectors = cosine_scorer.transform(all_token_documents)
        cosine_predictions: list[int] = []

        for index, (pair, left, right) in enumerate(
            zip(pairs, processed_a, processed_b)
        ):
            right_vector_index = index + len(processed_a)
            cosine_score = cosine_scorer.cosine(
                cosine_vectors[index],
                cosine_vectors[right_vector_index],
            )
            cosine_prediction = int(
                bool(left.tokens and right.tokens)
                and cosine_score >= cosine_threshold
            )
            cosine_predictions.append(cosine_prediction)
            lsh_details[index].update({
                "cosine_similarity": round(cosine_score, 6),
                "cosine_prediction": cosine_prediction,
                "error_cosine": int(cosine_prediction != pair.label),
            })

        cosine_runtime = perf_counter() - cosine_start
        cosine_metrics = classification_metrics(labels, cosine_predictions)
        metric_rows.append({
            "method": "TF-IDF cosine (bonus)",
            **{
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in cosine_metrics.as_dict().items()
            },
            "runtime_seconds": round(cosine_runtime, 6),
            "pairs_evaluated": total,
            "candidate_pairs": total,
            "candidate_recall": "",
            "comparison_reduction": 0.0,
            "jaccard_threshold": "",
            "simhash_max_distance": "",
            "cosine_threshold": cosine_threshold,
            "num_perm": "",
            "bands": "",
        })

    return metric_rows, lsh_details
