from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from plagiarism_engine.cli import main
from plagiarism_engine.dataset import LabeledPair
from plagiarism_engine.evaluation import classification_metrics, evaluate_pairs
from plagiarism_engine.lsh import LSHIndex
from plagiarism_engine.minhash import MinHasher, exact_jaccard
from plagiarism_engine.preprocessing import TextPreprocessor
from plagiarism_engine.simhash import TfidfSimHasher, hamming_distance
from plagiarism_engine.tfidf_cosine import TfidfCosineScorer


class PreprocessingTests(unittest.TestCase):
    def test_english_normalization_stopwords_and_shingles(self) -> None:
        processor = TextPreprocessor(shingle_size=3)
        result = processor.preprocess("The QUICK, brown fox jumps over the lazy dog!")
        self.assertEqual(result.normalized_text, "the quick brown fox jumps over the lazy dog")
        self.assertEqual(result.tokens, ("quick", "brown", "fox", "jumps", "over", "lazy", "dog"))
        self.assertIn(("quick", "brown", "fox"), result.shingles)

    def test_persian_character_normalization(self) -> None:
        processor = TextPreprocessor(language="fa")
        result = processor.preprocess("اين يك متنِ آزمايشي است.")
        self.assertIn("این", result.normalized_text)
        self.assertIn("آزمایشی", result.normalized_text)
        self.assertNotIn("یک", result.tokens)

    def test_short_and_empty_text(self) -> None:
        processor = TextPreprocessor(shingle_size=3)
        self.assertEqual(processor.preprocess("!!!").shingles, frozenset())
        self.assertEqual(processor.preprocess("rare words").shingles, frozenset({("rare", "words")}))

    def test_adaptive_short_text_shingles(self) -> None:
        processor = TextPreprocessor(
            shingle_size=3,
            adaptive_short_text=True,
        )
        result = processor.preprocess("how can I learn python quickly")
        self.assertIn(("learn",), result.shingles)
        self.assertIn(("learn", "python"), result.shingles)



class SimilarityTests(unittest.TestCase):
    def test_exact_jaccard(self) -> None:
        self.assertAlmostEqual(exact_jaccard({1, 2, 3}, {2, 3, 4}), 0.5)
        self.assertEqual(exact_jaccard(set(), set()), 0.0)

    def test_minhash_identical_and_disjoint(self) -> None:
        hasher = MinHasher(num_perm=128, seed=7)
        same_a = hasher.signature({("a", "b", "c"), ("b", "c", "d")})
        same_b = hasher.signature({("a", "b", "c"), ("b", "c", "d")})
        disjoint = hasher.signature({("x", "y", "z")})
        self.assertEqual(hasher.similarity(same_a, same_b), 1.0)
        self.assertLess(hasher.similarity(same_a, disjoint), 0.2)

    def test_lsh_candidate(self) -> None:
        hasher = MinHasher(num_perm=128, seed=1)
        sig_a = hasher.signature({("one", "two", "three"), ("two", "three", "four")})
        sig_b = hasher.signature({("one", "two", "three"), ("two", "three", "four")})
        index = LSHIndex(num_perm=128, bands=64)
        index.insert("a", sig_a)
        index.insert("b", sig_b)
        self.assertIn(("a", "b"), index.candidate_pairs())
        self.assertTrue(index.is_candidate(sig_a, sig_b))

    def test_simhash_identical_closer_than_unrelated(self) -> None:
        docs = [
            ("machine", "learning", "models", "learn", "patterns"),
            ("machine", "learning", "models", "learn", "patterns"),
            ("fresh", "bread", "needs", "flour", "yeast"),
        ]
        model = TfidfSimHasher().fit(docs)
        fingerprints = model.fingerprints(docs)
        self.assertEqual(hamming_distance(fingerprints[0], fingerprints[1]), 0)
        self.assertLess(
            hamming_distance(fingerprints[0], fingerprints[1]),
            hamming_distance(fingerprints[0], fingerprints[2]),
        )

    def test_cosine_similar_texts_score_higher(self) -> None:
        documents = [
            ("how", "learn", "python"),
            ("best", "way", "learn", "python"),
            ("bread", "flour", "oven"),
        ]
        scorer = TfidfCosineScorer().fit(documents)
        vectors = scorer.transform(documents)
        similar_score = scorer.cosine(vectors[0], vectors[1])
        unrelated_score = scorer.cosine(vectors[0], vectors[2])
        self.assertGreater(similar_score, unrelated_score)



class EvaluationTests(unittest.TestCase):
    def test_metrics(self) -> None:
        metrics = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((metrics.tp, metrics.fp, metrics.tn, metrics.fn), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics.f1, 0.5)

    def test_pair_evaluation_outputs_two_methods(self) -> None:
        pairs = [
            LabeledPair("1", "alpha beta gamma delta", "alpha beta gamma delta", 1),
            LabeledPair("2", "alpha beta gamma delta", "bread flour yeast oven", 0),
        ]
        metrics, predictions = evaluate_pairs(
            pairs,
            preprocessor=TextPreprocessor(),
            minhasher=MinHasher(128, 42),
            bands=64,
            jaccard_threshold=0.25,
            simhash_max_distance=18,
        )
        self.assertEqual(len(metrics), 2)
        self.assertEqual(len(predictions), 2)
        self.assertIn("candidate_recall", metrics[0])

    def test_pair_evaluation_with_cosine(self) -> None:
        pairs = [
            LabeledPair(
                "1",
                "how can I learn python quickly",
                "best way to learn python quickly",
                1,
            ),
            LabeledPair(
                "2",
                "how can I learn python quickly",
                "bread needs flour and yeast",
                0,
            ),
        ]
        metrics, predictions = evaluate_pairs(
            pairs,
            preprocessor=TextPreprocessor(adaptive_short_text=True),
            minhasher=MinHasher(128, 42),
            bands=64,
            jaccard_threshold=0.05,
            simhash_max_distance=22,
            simhash_mode="hybrid",
            cosine_threshold=0.20,
        )
        self.assertEqual(len(metrics), 3)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(metrics[2]["method"], "TF-IDF cosine (bonus)")
        self.assertIn("cosine_similarity", predictions[0])



class CliTests(unittest.TestCase):
    def test_compare_command_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / "a.txt"
            b = root / "b.txt"
            output = root / "result.json"
            a.write_text("machine learning finds patterns in data", encoding="utf-8")
            b.write_text("machine learning finds useful patterns in data", encoding="utf-8")
            code = main([
                "compare", "--file-a", str(a), "--file-b", str(b),
                "--output", str(output),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("minhash_lsh", payload)
            self.assertIn("simhash", payload)


if __name__ == "__main__":
    unittest.main()
