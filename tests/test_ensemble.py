"""Tests for the calibrated lexical ensemble bonus model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plagiarism_engine.cli import main
from plagiarism_engine.dataset import LabeledPair
from plagiarism_engine.ensemble import (
    CalibratedEnsemble,
    LinearEnsembleClassifier,
    PairFeatureExtractor,
    stratified_split_indices,
)
from plagiarism_engine.preprocessing import TextPreprocessor


class EnsembleTests(unittest.TestCase):
    def test_stratified_split_is_deterministic_and_disjoint(self) -> None:
        pairs = [
            LabeledPair(str(index), f"left {index}", f"right {index}", index % 2)
            for index in range(30)
        ]
        first = stratified_split_indices(pairs, seed=7)
        second = stratified_split_indices(pairs, seed=7)
        self.assertEqual(first, second)
        train, validation, test = map(set, first)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(len(train | validation | test), len(pairs))

    def test_feature_extractor_and_classifier(self) -> None:
        preprocessor = TextPreprocessor(adaptive_short_text=True)
        documents = [
            preprocessor.preprocess("how can I learn python quickly"),
            preprocessor.preprocess("what is the fastest way to learn python"),
            preprocessor.preprocess("bread needs flour and yeast"),
            preprocessor.preprocess("how do I bake bread"),
        ]
        extractor = PairFeatureExtractor().fit(doc.tokens for doc in documents)
        similar = extractor.features(documents[0], documents[1])
        unrelated = extractor.features(documents[0], documents[2])
        self.assertGreater(similar[0], unrelated[0])

        classifier = LinearEnsembleClassifier(l2=1.0).fit(
            np.asarray([similar, unrelated, similar * 0.95, unrelated * 0.8]),
            [1, 0, 1, 0],
            max_iter=15,
        )
        probabilities = classifier.predict_proba([similar, unrelated])
        self.assertGreater(probabilities[0], probabilities[1])

    def test_calibrate_cli_writes_loadable_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "pairs.csv"
            metrics_path = root / "metrics.csv"
            predictions_path = root / "predictions.csv"
            model_path = root / "ensemble.json"

            positives = [
                (f"how can I learn topic {index}", f"best way to learn topic {index}")
                for index in range(10)
            ]
            negatives = [
                (f"how can I learn topic {index}", f"bread flour oven recipe {index}")
                for index in range(10)
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "question1", "question2", "is_duplicate"])
                for index, (left, right) in enumerate(positives):
                    writer.writerow([f"p{index}", left, right, 1])
                for index, (left, right) in enumerate(negatives):
                    writer.writerow([f"n{index}", left, right, 0])

            status = main([
                "calibrate",
                "--pairs", str(csv_path),
                "--text-col-a", "question1",
                "--text-col-b", "question2",
                "--label-col", "is_duplicate",
                "--id-col", "id",
                "--adaptive-short-text",
                "--output", str(metrics_path),
                "--predictions-output", str(predictions_path),
                "--model-output", str(model_path),
            ])
            self.assertEqual(status, 0)
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(predictions_path.is_file())
            model = CalibratedEnsemble.load(model_path)
            score = model.score_texts(
                "how can I learn python",
                "best way to learn python",
            )
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            payload = json.loads(model_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_type"], "calibrated_lexical_ensemble")


if __name__ == "__main__":
    unittest.main()
