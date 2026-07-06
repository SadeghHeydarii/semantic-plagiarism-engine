"""Unit tests for the bonus algorithms and bonus-eval CLI command."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import csv

from plagiarism_engine.bonus import find_adaptive_lsh_params, HybridTfidfSimHasher, persian_lemmatize
from plagiarism_engine.cli import main


class TestBonusFeatures(unittest.TestCase):

    def test_find_adaptive_lsh_params(self) -> None:
        # For 128 permutations and threshold 0.15:
        # Divisors of 128: 1, 2, 4, 8, 16, 32, 64, 128
        # b=64, r=2 -> thresh = (1/64)^(1/2) = 0.125
        # b=128, r=1 -> thresh = (1/128)^1 = 0.0078
        # b=32, r=4 -> thresh = (1/32)^(1/4) = 0.42
        # So b=64 is closest to 0.15.
        b, r = find_adaptive_lsh_params(128, 0.15)
        self.assertEqual(b, 64)
        self.assertEqual(r, 2)

    def test_persian_lemmatizer(self) -> None:
        tokens = ("کتاب‌ها", "می‌روند", "می", "درختان", "اطلاعات")
        lemmas = persian_lemmatize(tokens)
        self.assertEqual(lemmas, ("کتاب", "می‌روند", "درخت", "اطلاع"))

    def test_hybrid_simhash(self) -> None:
        hasher = HybridTfidfSimHasher(use_char_ngrams=True)
        hasher.fit([["quick", "fox"], ["lazy", "dog"]])
        fp = hasher.fingerprint(["quick", "fox"])
        self.assertIsInstance(fp, int)
        self.assertNotEqual(fp, 0)

    def test_bonus_eval_command(self) -> None:
        # Create a mock CSV pair dataset in a temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_pairs.csv"
            out_path = Path(tmpdir) / "bonus_metrics.csv"
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "question1", "question2", "is_duplicate"])
                writer.writerow(["1", "what is the difference?", "what are differences?", "1"])
                writer.writerow(["2", "how to learn data mining?", "where can I learn python?", "0"])
            
            # Execute main CLI command with bonus-eval
            status = main([
                "bonus-eval",
                "--pairs", str(csv_path),
                "--text-col-a", "question1",
                "--text-col-b", "question2",
                "--label-col", "is_duplicate",
                "--id-col", "id",
                "--limit", "10",
                "--output", str(out_path),
                "--num-perm", "128",
                "--bands", "64",
                "--jaccard-threshold", "0.15",
                "--simhash-max-distance", "18",
            ])
            self.assertEqual(status, 0)
            self.assertTrue(out_path.is_file())


if __name__ == "__main__":
    unittest.main()
