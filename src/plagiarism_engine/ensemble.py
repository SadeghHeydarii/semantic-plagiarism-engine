"""Calibrated lexical ensemble for duplicate-question and near-plagiarism detection.

The original project pipelines remain unchanged.  This module adds a transparent
bonus model that combines several lexical similarity signals and learns a small
logistic-regression layer from labeled pairs.  The optimization is implemented
locally with NumPy (no scikit-learn or ready-made classifier is used).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .dataset import LabeledPair
from .preprocessing import PreprocessedDocument, TextPreprocessor
from .tfidf_cosine import TfidfCosineScorer


QUESTION_WORDS = frozenset({
    "what", "why", "how", "when", "where", "who", "which",
    "can", "should", "is", "are", "do", "does", "did",
    "will", "would", "could",
})

NEGATIONS = frozenset({
    "not", "no", "never", "none", "cannot", "cant", "don't",
    "dont", "doesnt", "didnt", "wont", "wouldnt", "shouldnt",
    "couldnt",
})

BASE_FEATURE_NAMES: tuple[str, ...] = (
    "word_tfidf_cosine",
    "word_bigram_tfidf_cosine",
    "char_tfidf_cosine",
    "token_jaccard",
    "overlap_coefficient",
    "length_ratio",
    "question_word_same",
    "question_word_mismatch",
    "negation_mismatch",
    "sequence_ratio",
    "sorted_token_ratio",
    "bigram_jaccard",
    "first_token_same",
    "common_prefix_ratio",
    "exact_match",
)

_CONTINUOUS_SQUARE_INDICES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 9, 10, 11, 13,
)

_INTERACTION_INDICES: tuple[tuple[int, int], ...] = (
    (0, 4), (2, 4), (3, 4), (9, 4), (10, 4),
    (0, 5), (2, 5), (3, 5), (9, 5),
    (0, 2), (0, 9), (2, 9), (3, 9), (4, 9),
)


def _expanded_feature_names() -> tuple[str, ...]:
    names = list(BASE_FEATURE_NAMES)
    names.extend(f"{BASE_FEATURE_NAMES[index]}^2" for index in _CONTINUOUS_SQUARE_INDICES)
    names.extend(
        f"{BASE_FEATURE_NAMES[left]}*{BASE_FEATURE_NAMES[right]}"
        for left, right in _INTERACTION_INDICES
    )
    return tuple(names)


EXPANDED_FEATURE_NAMES = _expanded_feature_names()


def _scorer_to_dict(scorer: TfidfCosineScorer) -> dict[str, object]:
    return {
        "word_ngram_sizes": list(scorer.word_ngram_sizes),
        "char_ngram_sizes": list(scorer.char_ngram_sizes),
        "document_count": scorer.document_count,
        "idf": scorer.idf,
    }


def _scorer_from_dict(payload: dict[str, object]) -> TfidfCosineScorer:
    scorer = TfidfCosineScorer(
        word_ngram_sizes=tuple(int(value) for value in payload["word_ngram_sizes"]),
        char_ngram_sizes=tuple(int(value) for value in payload["char_ngram_sizes"]),
    )
    scorer.document_count = int(payload["document_count"])
    scorer.idf = {
        str(feature): float(value)
        for feature, value in dict(payload["idf"]).items()
    }
    return scorer


class PairFeatureExtractor:
    """Extract deterministic similarity features for a pair of documents."""

    def __init__(self) -> None:
        self.word_scorer = TfidfCosineScorer(
            word_ngram_sizes=(1,),
            char_ngram_sizes=(),
        )
        self.word_bigram_scorer = TfidfCosineScorer(
            word_ngram_sizes=(1, 2),
            char_ngram_sizes=(),
        )
        self.char_scorer = TfidfCosineScorer(
            word_ngram_sizes=(),
            char_ngram_sizes=(3, 4, 5),
        )
        self.is_fitted = False

    def fit(
        self,
        token_documents: Iterable[Sequence[str]],
    ) -> "PairFeatureExtractor":
        documents = [tuple(tokens) for tokens in token_documents]
        if not documents:
            raise ValueError("At least one training document is required.")
        self.word_scorer.fit(documents)
        self.word_bigram_scorer.fit(documents)
        self.char_scorer.fit(documents)
        self.is_fitted = True
        return self

    @staticmethod
    def _cosine_pair(
        scorer: TfidfCosineScorer,
        left: Sequence[str],
        right: Sequence[str],
    ) -> float:
        return scorer.cosine(
            scorer.vector(left),
            scorer.vector(right),
        )

    def base_features(
        self,
        left: PreprocessedDocument,
        right: PreprocessedDocument,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PairFeatureExtractor must be fitted first.")

        left_tokens = left.tokens
        right_tokens = right.tokens
        left_set = set(left_tokens)
        right_set = set(right_tokens)
        intersection_size = len(left_set & right_set)
        union_size = len(left_set | right_set)

        token_jaccard = intersection_size / union_size if union_size else 0.0
        overlap = (
            intersection_size / min(len(left_set), len(right_set))
            if left_set and right_set
            else 0.0
        )
        length_ratio = (
            min(len(left_tokens), len(right_tokens))
            / max(len(left_tokens), len(right_tokens))
            if left_tokens and right_tokens
            else 0.0
        )

        left_question = next(
            (token for token in left_tokens if token in QUESTION_WORDS),
            "",
        )
        right_question = next(
            (token for token in right_tokens if token in QUESTION_WORDS),
            "",
        )
        question_same = float(
            bool(left_question and left_question == right_question)
        )
        question_mismatch = float(
            bool(
                left_question
                and right_question
                and left_question != right_question
            )
        )
        negation_mismatch = float(
            bool(left_set & NEGATIONS) != bool(right_set & NEGATIONS)
        )

        sequence_ratio = SequenceMatcher(
            None,
            left.normalized_text,
            right.normalized_text,
            autojunk=False,
        ).ratio()
        sorted_token_ratio = SequenceMatcher(
            None,
            " ".join(sorted(left_tokens)),
            " ".join(sorted(right_tokens)),
            autojunk=False,
        ).ratio()

        left_bigrams = set(zip(left_tokens, left_tokens[1:]))
        right_bigrams = set(zip(right_tokens, right_tokens[1:]))
        bigram_union = left_bigrams | right_bigrams
        bigram_jaccard = (
            len(left_bigrams & right_bigrams) / len(bigram_union)
            if bigram_union
            else 0.0
        )

        common_prefix = 0
        for left_token, right_token in zip(left_tokens, right_tokens):
            if left_token != right_token:
                break
            common_prefix += 1
        common_prefix_ratio = (
            common_prefix / min(len(left_tokens), len(right_tokens))
            if left_tokens and right_tokens
            else 0.0
        )

        return np.asarray([
            self._cosine_pair(self.word_scorer, left_tokens, right_tokens),
            self._cosine_pair(
                self.word_bigram_scorer,
                left_tokens,
                right_tokens,
            ),
            self._cosine_pair(self.char_scorer, left_tokens, right_tokens),
            token_jaccard,
            overlap,
            length_ratio,
            question_same,
            question_mismatch,
            negation_mismatch,
            sequence_ratio,
            sorted_token_ratio,
            bigram_jaccard,
            float(
                bool(
                    left_tokens
                    and right_tokens
                    and left_tokens[0] == right_tokens[0]
                )
            ),
            common_prefix_ratio,
            float(
                bool(left.normalized_text)
                and left.normalized_text == right.normalized_text
            ),
        ], dtype=float)

    @staticmethod
    def expand_features(base_features: Sequence[float]) -> np.ndarray:
        base = np.asarray(base_features, dtype=float)
        if base.shape != (len(BASE_FEATURE_NAMES),):
            raise ValueError(
                f"Expected {len(BASE_FEATURE_NAMES)} base features, "
                f"received shape {base.shape}."
            )
        squared = base[list(_CONTINUOUS_SQUARE_INDICES)] ** 2
        interactions = np.asarray([
            base[left] * base[right]
            for left, right in _INTERACTION_INDICES
        ])
        return np.concatenate((base, squared, interactions))

    def features(
        self,
        left: PreprocessedDocument,
        right: PreprocessedDocument,
    ) -> np.ndarray:
        return self.expand_features(self.base_features(left, right))

    def to_dict(self) -> dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted feature extractor.")
        return {
            "word_scorer": _scorer_to_dict(self.word_scorer),
            "word_bigram_scorer": _scorer_to_dict(self.word_bigram_scorer),
            "char_scorer": _scorer_to_dict(self.char_scorer),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PairFeatureExtractor":
        extractor = cls()
        extractor.word_scorer = _scorer_from_dict(
            dict(payload["word_scorer"])
        )
        extractor.word_bigram_scorer = _scorer_from_dict(
            dict(payload["word_bigram_scorer"])
        )
        extractor.char_scorer = _scorer_from_dict(
            dict(payload["char_scorer"])
        )
        extractor.is_fitted = True
        return extractor


@dataclass(slots=True)
class LinearEnsembleClassifier:
    """Small logistic-regression classifier trained by Newton/IRLS updates."""

    l2: float = 10.0 / 3.0
    threshold: float = 0.5
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    weights: np.ndarray | None = None
    iterations: int = 0

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        *,
        max_iter: int = 25,
        tolerance: float = 1e-7,
        balanced: bool = True,
    ) -> "LinearEnsembleClassifier":
        matrix = np.asarray(features, dtype=float)
        target = np.asarray(labels, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("features must be a non-empty two-dimensional matrix.")
        if matrix.shape[0] != target.shape[0]:
            raise ValueError("features and labels must have equal row counts.")
        if set(np.unique(target)) - {0.0, 1.0}:
            raise ValueError("labels must contain only 0 and 1.")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative.")

        self.means = matrix.mean(axis=0)
        self.scales = matrix.std(axis=0)
        self.scales[self.scales < 1e-12] = 1.0
        standardized = (matrix - self.means) / self.scales
        design = np.column_stack((np.ones(matrix.shape[0]), standardized))

        sample_weights = np.ones(matrix.shape[0], dtype=float)
        if balanced:
            positives = float(target.sum())
            negatives = float(matrix.shape[0] - positives)
            if positives and negatives:
                sample_weights[target == 1.0] = matrix.shape[0] / (2.0 * positives)
                sample_weights[target == 0.0] = matrix.shape[0] / (2.0 * negatives)

        weights = np.zeros(design.shape[1], dtype=float)
        regularizer = np.eye(design.shape[1], dtype=float) * self.l2
        regularizer[0, 0] = 0.0

        for iteration in range(max_iter):
            logits = np.clip(design @ weights, -35.0, 35.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            variance = sample_weights * probabilities * (1.0 - probabilities)
            gradient = design.T @ (sample_weights * (target - probabilities))
            gradient[1:] -= self.l2 * weights[1:]
            information = (design.T * variance) @ design + regularizer
            try:
                delta = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(information, gradient, rcond=None)[0]
            weights += delta
            self.iterations = iteration + 1
            if float(np.max(np.abs(delta))) < tolerance:
                break

        self.weights = weights
        return self

    def predict_proba(
        self,
        features: Sequence[Sequence[float]],
    ) -> np.ndarray:
        if self.means is None or self.scales is None or self.weights is None:
            raise RuntimeError("LinearEnsembleClassifier must be fitted first.")
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        standardized = (matrix - self.means) / self.scales
        design = np.column_stack((np.ones(matrix.shape[0]), standardized))
        logits = np.clip(design @ self.weights, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(int)

    def to_dict(self) -> dict[str, object]:
        if self.means is None or self.scales is None or self.weights is None:
            raise RuntimeError("Cannot serialize an unfitted classifier.")
        return {
            "l2": self.l2,
            "threshold": self.threshold,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "weights": self.weights.tolist(),
            "iterations": self.iterations,
            "feature_names": list(EXPANDED_FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LinearEnsembleClassifier":
        classifier = cls(
            l2=float(payload["l2"]),
            threshold=float(payload["threshold"]),
        )
        classifier.means = np.asarray(payload["means"], dtype=float)
        classifier.scales = np.asarray(payload["scales"], dtype=float)
        classifier.weights = np.asarray(payload["weights"], dtype=float)
        classifier.iterations = int(payload.get("iterations", 0))
        return classifier


@dataclass(slots=True)
class CalibratedEnsemble:
    extractor: PairFeatureExtractor
    classifier: LinearEnsembleClassifier
    preprocessing: dict[str, object]

    def score_documents(
        self,
        left: PreprocessedDocument,
        right: PreprocessedDocument,
    ) -> float:
        features = self.extractor.features(left, right)
        return float(self.classifier.predict_proba([features])[0])

    def predict_documents(
        self,
        left: PreprocessedDocument,
        right: PreprocessedDocument,
    ) -> int:
        return int(self.score_documents(left, right) >= self.classifier.threshold)

    def make_preprocessor(self) -> TextPreprocessor:
        return TextPreprocessor(**self.preprocessing)

    def score_texts(self, left_text: str, right_text: str) -> float:
        preprocessor = self.make_preprocessor()
        return self.score_documents(
            preprocessor.preprocess(left_text),
            preprocessor.preprocess(right_text),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "model_type": "calibrated_lexical_ensemble",
            "preprocessing": self.preprocessing,
            "extractor": self.extractor.to_dict(),
            "classifier": self.classifier.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CalibratedEnsemble":
        if payload.get("model_type") != "calibrated_lexical_ensemble":
            raise ValueError("Unsupported ensemble model file.")
        return cls(
            extractor=PairFeatureExtractor.from_dict(dict(payload["extractor"])),
            classifier=LinearEnsembleClassifier.from_dict(
                dict(payload["classifier"])
            ),
            preprocessing=dict(payload["preprocessing"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedEnsemble":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def preprocessor_config(preprocessor: TextPreprocessor) -> dict[str, object]:
    return {
        "shingle_size": preprocessor.shingle_size,
        "language": preprocessor.language,
        "remove_stopwords": preprocessor.remove_stopwords,
        "short_text_policy": preprocessor.short_text_policy,
        "adaptive_short_text": preprocessor.adaptive_short_text,
        "short_text_token_limit": preprocessor.short_text_token_limit,
    }


def stratified_split_indices(
    pairs: Sequence[LabeledPair],
    *,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    groups: dict[int, list[int]] = {0: [], 1: []}
    for index, pair in enumerate(pairs):
        groups[pair.label].append(index)

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for label, indices in groups.items():
        indices.sort(
            key=lambda index: hashlib.blake2b(
                f"{seed}:{label}:{pairs[index].pair_id}:{index}".encode("utf-8"),
                digest_size=8,
                person=b"plag-split",
            ).digest()
        )
        train_end = int(len(indices) * train_ratio)
        validation_end = train_end + int(len(indices) * validation_ratio)
        train.extend(indices[:train_end])
        validation.extend(indices[train_end:validation_end])
        test.extend(indices[validation_end:])

    return sorted(train), sorted(validation), sorted(test)


def binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, float | int]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty and equal length.")
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def tune_f1_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[float, dict[str, float | int]]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and equal length.")

    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    total_positive = sum(labels)
    tp = 0
    fp = 0
    best_threshold = 1.0
    best_metrics = binary_metrics(labels, [0] * len(labels))

    cursor = 0
    while cursor < len(order):
        score = float(scores[order[cursor]])
        next_cursor = cursor
        while next_cursor < len(order) and float(scores[order[next_cursor]]) == score:
            if labels[order[next_cursor]] == 1:
                tp += 1
            else:
                fp += 1
            next_cursor += 1
        fn = total_positive - tp
        tn = len(labels) - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(labels)
        candidate = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        if (
            candidate["f1"] > best_metrics["f1"]
            or (
                math.isclose(float(candidate["f1"]), float(best_metrics["f1"]))
                and candidate["accuracy"] > best_metrics["accuracy"]
            )
        ):
            best_threshold = score
            best_metrics = candidate
        cursor = next_cursor

    return best_threshold, best_metrics


def calibrate_ensemble(
    pairs: Sequence[LabeledPair],
    *,
    preprocessor: TextPreprocessor,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    seed: int = 42,
    l2: float = 10.0 / 3.0,
    max_iter: int = 25,
) -> tuple[
    CalibratedEnsemble,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, int],
]:
    if len(pairs) < 10:
        raise ValueError("At least 10 labeled pairs are required for calibration.")

    train_indices, validation_indices, test_indices = stratified_split_indices(
        pairs,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    if not train_indices or not validation_indices or not test_indices:
        raise ValueError("The selected split produced an empty partition.")

    processed_a = [preprocessor.preprocess(pair.text_a) for pair in pairs]
    processed_b = [preprocessor.preprocess(pair.text_b) for pair in pairs]

    training_documents = (
        [processed_a[index].tokens for index in train_indices]
        + [processed_b[index].tokens for index in train_indices]
    )
    extractor = PairFeatureExtractor().fit(training_documents)

    feature_matrix = np.vstack([
        extractor.features(left, right)
        for left, right in zip(processed_a, processed_b)
    ])
    labels = np.asarray([pair.label for pair in pairs], dtype=int)

    classifier = LinearEnsembleClassifier(l2=l2)
    classifier.fit(
        feature_matrix[train_indices],
        labels[train_indices],
        max_iter=max_iter,
        balanced=True,
    )
    validation_scores = classifier.predict_proba(feature_matrix[validation_indices])
    threshold, validation_metrics = tune_f1_threshold(
        labels[validation_indices].tolist(),
        validation_scores.tolist(),
    )
    classifier.threshold = threshold

    test_scores = classifier.predict_proba(feature_matrix[test_indices])
    test_predictions = (test_scores >= threshold).astype(int)
    test_metrics = binary_metrics(
        labels[test_indices].tolist(),
        test_predictions.tolist(),
    )

    # Two transparent baselines on the same held-out split.
    char_validation_scores = feature_matrix[validation_indices, 2]
    char_threshold, char_validation_metrics = tune_f1_threshold(
        labels[validation_indices].tolist(),
        char_validation_scores.tolist(),
    )
    char_test_predictions = (
        feature_matrix[test_indices, 2] >= char_threshold
    ).astype(int)
    char_test_metrics = binary_metrics(
        labels[test_indices].tolist(),
        char_test_predictions.tolist(),
    )

    jaccard_validation_scores = feature_matrix[validation_indices, 3]
    jaccard_threshold, jaccard_validation_metrics = tune_f1_threshold(
        labels[validation_indices].tolist(),
        jaccard_validation_scores.tolist(),
    )
    jaccard_test_predictions = (
        feature_matrix[test_indices, 3] >= jaccard_threshold
    ).astype(int)
    jaccard_test_metrics = binary_metrics(
        labels[test_indices].tolist(),
        jaccard_test_predictions.tolist(),
    )

    split_counts = {
        "train_pairs": len(train_indices),
        "validation_pairs": len(validation_indices),
        "test_pairs": len(test_indices),
    }

    def metric_row(
        method: str,
        split: str,
        metrics: dict[str, float | int],
        selected_threshold: float,
    ) -> dict[str, object]:
        return {
            "method": method,
            "split": split,
            **{
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in metrics.items()
            },
            "threshold": round(selected_threshold, 6),
            **split_counts,
            "feature_count": len(EXPANDED_FEATURE_NAMES),
            "l2": round(l2, 6),
        }

    metric_rows = [
        metric_row(
            "Character TF-IDF cosine",
            "validation",
            char_validation_metrics,
            char_threshold,
        ),
        metric_row(
            "Character TF-IDF cosine",
            "test",
            char_test_metrics,
            char_threshold,
        ),
        metric_row(
            "Token Jaccard",
            "validation",
            jaccard_validation_metrics,
            jaccard_threshold,
        ),
        metric_row(
            "Token Jaccard",
            "test",
            jaccard_test_metrics,
            jaccard_threshold,
        ),
        metric_row(
            "Calibrated lexical ensemble",
            "validation",
            validation_metrics,
            threshold,
        ),
        metric_row(
            "Calibrated lexical ensemble",
            "test",
            test_metrics,
            threshold,
        ),
    ]

    prediction_rows: list[dict[str, object]] = []
    for local_index, pair_index in enumerate(test_indices):
        base = extractor.base_features(
            processed_a[pair_index],
            processed_b[pair_index],
        )
        score = float(test_scores[local_index])
        prediction = int(test_predictions[local_index])
        prediction_rows.append({
            "pair_id": pairs[pair_index].pair_id,
            "label": pairs[pair_index].label,
            "ensemble_score": round(score, 6),
            "ensemble_prediction": prediction,
            "error_ensemble": int(prediction != pairs[pair_index].label),
            "word_tfidf_cosine": round(float(base[0]), 6),
            "char_tfidf_cosine": round(float(base[2]), 6),
            "token_jaccard": round(float(base[3]), 6),
            "overlap_coefficient": round(float(base[4]), 6),
            "sequence_ratio": round(float(base[9]), 6),
            "text_a": pairs[pair_index].text_a,
            "text_b": pairs[pair_index].text_b,
        })

    model = CalibratedEnsemble(
        extractor=extractor,
        classifier=classifier,
        preprocessing=preprocessor_config(preprocessor),
    )
    return model, metric_rows, prediction_rows, split_counts
