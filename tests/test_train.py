"""
tests/test_train.py
-------------------
Unit tests for autoresearch/train.py helpers.

Tests cover:
  - _append_results(): header written on fresh file
  - _append_results(): header written when file exists but is empty (stale)
  - _append_results(): NO header written on subsequent appends to non-empty file
  - _append_results(): creates parent directory if missing
  - _loss_to_score(): normalises finite / infinite / NaN loss correctly
  - TrainConfig: default field values are sensible
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoresearch.train import _append_results, _loss_to_score, TrainConfig


# ── _append_results ───────────────────────────────────────────────────────────

class TestAppendResults(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tsv = str(Path(self.tmp) / "results.tsv")

    def _read_tsv(self) -> list[dict]:
        with open(self.tsv, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def test_fresh_file_has_header(self):
        _append_results({"loss": 1.2, "score": 0.8}, self.tsv)
        rows = self._read_tsv()
        self.assertEqual(len(rows), 1)
        self.assertIn("loss", rows[0])
        self.assertIn("score", rows[0])

    def test_empty_existing_file_gets_header(self):
        """Regression: stale empty file from a previous run must still get a header."""
        Path(self.tsv).touch()                       # create empty file
        _append_results({"loss": 0.5, "score": 0.9}, self.tsv)
        rows = self._read_tsv()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["loss"], "0.5")

    def test_second_append_does_not_duplicate_header(self):
        """Two consecutive calls → 2 data rows, 1 header row."""
        _append_results({"loss": 1.0, "score": 0.7}, self.tsv)
        _append_results({"loss": 0.8, "score": 0.75}, self.tsv)
        rows = self._read_tsv()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["loss"], "1.0")
        self.assertEqual(rows[1]["loss"], "0.8")

    def test_creates_parent_directory(self):
        nested = str(Path(self.tmp) / "deep" / "results.tsv")
        _append_results({"x": 1}, nested)
        self.assertTrue(Path(nested).exists())

    def test_values_are_written_correctly(self):
        metrics = {"iteration": 3, "topic": "LoRA", "score": 0.92}
        _append_results(metrics, self.tsv)
        rows = self._read_tsv()
        self.assertEqual(rows[0]["iteration"], "3")
        self.assertEqual(rows[0]["topic"], "LoRA")
        self.assertEqual(rows[0]["score"], "0.92")


# ── _loss_to_score ────────────────────────────────────────────────────────────

class TestLossToScore(unittest.TestCase):

    def test_zero_loss_gives_one(self):
        self.assertAlmostEqual(_loss_to_score(0.0), 1.0)

    def test_loss_10_gives_zero(self):
        self.assertAlmostEqual(_loss_to_score(10.0), 0.0)

    def test_loss_5_gives_half(self):
        self.assertAlmostEqual(_loss_to_score(5.0), 0.5)

    def test_nan_loss_gives_zero(self):
        import math
        self.assertEqual(_loss_to_score(float("nan")), 0.0)

    def test_inf_loss_gives_zero(self):
        self.assertEqual(_loss_to_score(float("inf")), 0.0)

    def test_score_never_negative(self):
        # loss > 10 should clamp to 0, not go negative
        self.assertEqual(_loss_to_score(100.0), 0.0)


# ── TrainConfig defaults ──────────────────────────────────────────────────────

class TestTrainConfig(unittest.TestCase):

    def test_default_model_is_set(self):
        cfg = TrainConfig()
        self.assertIsInstance(cfg.model_name, str)
        self.assertGreater(len(cfg.model_name), 0)

    def test_default_epochs_is_positive(self):
        cfg = TrainConfig()
        self.assertGreater(cfg.num_train_epochs, 0)

    def test_default_learning_rate_is_positive(self):
        cfg = TrainConfig()
        self.assertGreater(cfg.learning_rate, 0)

    def test_override_fields(self):
        cfg = TrainConfig(num_train_epochs=5, learning_rate=1e-3)
        self.assertEqual(cfg.num_train_epochs, 5)
        self.assertAlmostEqual(cfg.learning_rate, 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
