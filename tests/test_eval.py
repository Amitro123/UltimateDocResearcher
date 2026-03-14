"""
tests/test_eval.py
------------------
Unit tests for autoresearch/eval.py — LLM-as-a-Judge evaluator.

Tests cover:
  - JudgeScores parsing from LLM text output
  - Heuristic fallback scoring
  - Full run_eval pipeline with a minimal in-memory val set
  - Output file creation and structure
  - Edge cases: empty val set, malformed records, missing fields
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoresearch.eval import (
    JudgeScores,
    SampleResult,
    _heuristic_score,
    _judge_prompt,
    run_eval,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_val_record(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "context": "test context",
    }


def _write_val_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


# ── JudgeScores parsing ───────────────────────────────────────────────────────

class TestJudgeScoresParsing(unittest.TestCase):

    def test_parses_well_formed_output(self):
        text = (
            "Accuracy: 4\n"
            "Relevance: 5\n"
            "Completeness: 3\n"
            "Reasoning: The answer covered the main points accurately."
        )
        scores = JudgeScores.from_parse(text)
        self.assertEqual(scores.accuracy, 4)
        self.assertEqual(scores.relevance, 5)
        self.assertEqual(scores.completeness, 3)
        self.assertAlmostEqual(scores.overall, 4 * 0.4 + 5 * 0.35 + 3 * 0.25, places=5)
        self.assertIn("covered", scores.reasoning)

    def test_parses_with_labels_mixed_case(self):
        text = "ACCURACY: 3\nRELEVANCE: 2\nCOMPLETENESS: 5\nREASONING: ok"
        scores = JudgeScores.from_parse(text)
        self.assertEqual(scores.accuracy, 3)
        self.assertEqual(scores.relevance, 2)
        self.assertEqual(scores.completeness, 5)

    def test_defaults_to_3_when_field_missing(self):
        text = "Accuracy: 5\nReasoning: good"
        scores = JudgeScores.from_parse(text)
        self.assertEqual(scores.relevance, 3)
        self.assertEqual(scores.completeness, 3)

    def test_is_passing_above_threshold(self):
        s = JudgeScores(accuracy=4, relevance=4, completeness=4,
                        reasoning="good", overall=4.0)
        self.assertTrue(s.is_passing(threshold=3.0))
        self.assertFalse(s.is_passing(threshold=4.5))

    def test_overall_clamped_to_valid_range(self):
        # Even if individual scores are extreme, formula stays in 1-5
        text = "Accuracy: 5\nRelevance: 5\nCompleteness: 5\nReasoning: perfect"
        scores = JudgeScores.from_parse(text)
        self.assertLessEqual(scores.overall, 5.0)
        self.assertGreaterEqual(scores.overall, 1.0)


# ── Heuristic scoring ─────────────────────────────────────────────────────────

class TestHeuristicScore(unittest.TestCase):

    def test_identical_texts_give_high_score(self):
        text = "LoRA adds low rank matrices to transformer layers for efficient fine tuning"
        scores = _heuristic_score(text, text)
        self.assertGreaterEqual(scores.overall, 4.0)

    def test_unrelated_texts_give_low_score(self):
        ref = "LoRA fine-tuning uses low-rank matrices to reduce trainable parameters"
        ans = "The weather in Paris is typically mild and rainy during spring"
        scores = _heuristic_score(ref, ans)
        self.assertLessEqual(scores.overall, 2.0)

    def test_empty_reference_returns_score_1(self):
        scores = _heuristic_score("", "some answer")
        self.assertEqual(scores.overall, 1.0)

    def test_partial_overlap_gives_mid_score(self):
        ref = "LoRA fine-tuning adds trainable rank decomposition matrices to frozen weights"
        ans = "LoRA adds matrices to frozen weights for fine-tuning"
        scores = _heuristic_score(ref, ans)
        self.assertGreater(scores.overall, 1.0)
        self.assertLess(scores.overall, 5.0)


# ── Judge prompt ──────────────────────────────────────────────────────────────

class TestJudgePrompt(unittest.TestCase):

    def test_prompt_contains_all_three_fields(self):
        prompt = _judge_prompt("What is LoRA?", "LoRA is a PEFT method.", "LoRA uses low-rank decomposition.")
        self.assertIn("What is LoRA?", prompt)
        self.assertIn("LoRA is a PEFT method.", prompt)
        self.assertIn("LoRA uses low-rank decomposition.", prompt)

    def test_long_inputs_are_truncated(self):
        long_text = "word " * 1000
        prompt = _judge_prompt("Q?", long_text, long_text)
        # Prompt should not be absurdly large
        self.assertLess(len(prompt), 5000)


# ── run_eval pipeline ─────────────────────────────────────────────────────────

class TestRunEval(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.val_path = Path(self.tmp) / "val.jsonl"
        self.output_dir = Path(self.tmp) / "results"

    def _write_val(self, records):
        _write_val_jsonl(self.val_path, records)

    def test_run_eval_produces_report(self):
        self._write_val([
            _make_val_record(
                "What is LoRA?",
                "LoRA uses low-rank matrices to reduce trainable parameters."
            ),
            _make_val_record(
                "What is chunking with overlap?",
                "Chunking splits text into overlapping windows for context preservation."
            ),
        ])
        report = run_eval(
            val_path=self.val_path,
            model_path=None,
            judge_model="gpt-4o-mini",  # will fall back to heuristic
            max_samples=10,
            output_dir=self.output_dir,
        )
        self.assertIn("summary", report)
        self.assertIn("samples", report)
        self.assertEqual(report["summary"]["n_samples"], 2)
        self.assertIn("pass_rate", report["summary"])
        self.assertIn("avg_overall", report["summary"])

    def test_report_json_is_written_to_disk(self):
        self._write_val([_make_val_record("Q?", "A.")])
        run_eval(
            val_path=self.val_path,
            judge_model="gpt-4o-mini",
            output_dir=self.output_dir,
        )
        report_path = self.output_dir / "eval_report.json"
        self.assertTrue(report_path.exists())
        data = json.loads(report_path.read_text())
        self.assertIn("summary", data)

    def test_max_samples_cap(self):
        records = [_make_val_record(f"Q{i}?", f"A{i}.") for i in range(20)]
        self._write_val(records)
        report = run_eval(
            val_path=self.val_path,
            judge_model="gpt-4o-mini",
            max_samples=5,
            output_dir=self.output_dir,
        )
        self.assertEqual(report["summary"]["n_samples"], 5)

    def test_empty_val_file_returns_empty(self):
        self.val_path.write_text("")
        report = run_eval(
            val_path=self.val_path,
            judge_model="gpt-4o-mini",
            output_dir=self.output_dir,
        )
        self.assertEqual(report, {})

    def test_malformed_record_is_skipped(self):
        self.val_path.write_text(
            json.dumps({"messages": []}) + "\n" +
            json.dumps(_make_val_record("Good Q?", "Good A.")) + "\n"
        )
        report = run_eval(
            val_path=self.val_path,
            judge_model="gpt-4o-mini",
            output_dir=self.output_dir,
        )
        # Only the valid record should be evaluated
        self.assertEqual(report["summary"]["n_samples"], 1)

    def test_sample_results_have_expected_keys(self):
        self._write_val([_make_val_record("What is PEFT?", "PEFT reduces trainable params.")])
        report = run_eval(
            val_path=self.val_path,
            judge_model="gpt-4o-mini",
            output_dir=self.output_dir,
        )
        sample = report["samples"][0]
        for key in ("question", "reference", "model_answer", "scores", "passed"):
            self.assertIn(key, sample)
        for key in ("accuracy", "relevance", "completeness", "overall"):
            self.assertIn(key, sample["scores"])

    def test_scores_are_in_valid_range(self):
        records = [_make_val_record(f"Q{i}?", f"A{i}.") for i in range(5)]
        self._write_val(records)
        report = run_eval(
            val_path=self.val_path,
            judge_model="gpt-4o-mini",
            output_dir=self.output_dir,
        )
        for s in report["samples"]:
            self.assertGreaterEqual(s["scores"]["overall"], 1.0)
            self.assertLessEqual(s["scores"]["overall"], 5.0)

    def test_pass_threshold_applied_correctly(self):
        self._write_val([_make_val_record("Q?", "A.")])
        # threshold=5.5 → nothing passes
        report = run_eval(
            val_path=self.val_path,
            judge_model="gpt-4o-mini",
            pass_threshold=5.5,
            output_dir=self.output_dir,
        )
        self.assertEqual(report["summary"]["pass_rate"], 0.0)

    def test_iteration_and_topic_in_summary(self):
        self._write_val([_make_val_record("Q?", "A.")])
        report = run_eval(
            val_path=self.val_path,
            judge_model="gpt-4o-mini",
            output_dir=self.output_dir,
            iteration=7,
            topic="LoRA fine-tuning",
        )
        self.assertEqual(report["summary"]["iteration"], 7)
        self.assertEqual(report["summary"]["topic"], "LoRA fine-tuning")


if __name__ == "__main__":
    unittest.main(verbosity=2)
