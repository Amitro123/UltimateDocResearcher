"""
tests/test_eval_spec.py
-----------------------
Tests for eval/eval_spec.yaml parsing and eval/run_eval.py logic.

Covers:
  - YAML loads without PyYAML (minimal parser)
  - YAML loads with PyYAML if available
  - All 5 expected criteria present with correct names
  - Weights are numeric and positive
  - Questions are non-empty strings
  - compute_weighted_score: correct math, edge cases
  - _parse_score: all formats parsed correctly
  - _heuristic_score: returns int 1-5 for each criterion
  - evaluate(): full pipeline with mocked LLM + temp file
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPEC_PATH = Path(__file__).resolve().parent.parent / "eval" / "eval_spec.yaml"


# ── YAML loading ──────────────────────────────────────────────────────────────

class TestLoadSpec(unittest.TestCase):

    def test_spec_file_exists(self):
        self.assertTrue(SPEC_PATH.exists(), f"eval_spec.yaml not found at {SPEC_PATH}")

    def test_loads_6_criteria(self):
        from eval.run_eval import load_spec
        criteria = load_spec(SPEC_PATH)
        self.assertEqual(len(criteria), 6, f"Expected 6 criteria, got {len(criteria)}")

    def test_expected_criterion_names(self):
        from eval.run_eval import load_spec
        criteria = load_spec(SPEC_PATH)
        names = {c["name"] for c in criteria}
        expected = {"clarity", "completeness", "actionability", "freshness", "anti_patterns", "external_novelty"}
        self.assertEqual(names, expected)

    def test_all_criteria_have_questions(self):
        from eval.run_eval import load_spec
        criteria = load_spec(SPEC_PATH)
        for c in criteria:
            self.assertIn("question", c, f"Criterion '{c.get('name')}' missing question")
            self.assertTrue(c["question"].strip(), f"Criterion '{c.get('name')}' has empty question")

    def test_all_weights_are_positive_numbers(self):
        from eval.run_eval import load_spec
        criteria = load_spec(SPEC_PATH)
        for c in criteria:
            self.assertIn("weight", c, f"Criterion '{c.get('name')}' missing weight")
            self.assertIsInstance(c["weight"], (int, float))
            self.assertGreater(c["weight"], 0)

    def test_clarity_has_highest_weight(self):
        from eval.run_eval import load_spec
        criteria = load_spec(SPEC_PATH)
        weights = {c["name"]: c["weight"] for c in criteria}
        self.assertEqual(
            max(weights, key=weights.get), "clarity",
            "clarity should have the highest weight"
        )

    def test_load_yaml_uses_pyyaml(self):
        """_load_yaml should successfully load the spec via PyYAML."""
        import yaml  # pyyaml is in requirements.txt — must be present
        from eval.run_eval import _load_yaml
        data = _load_yaml(SPEC_PATH)
        self.assertIn("criteria", data)
        self.assertIsInstance(data["criteria"], list)
        self.assertGreater(len(data["criteria"]), 0)


# ── compute_weighted_score ────────────────────────────────────────────────────

class TestComputeWeightedScore(unittest.TestCase):

    def _make_criteria(self, names_weights):
        return [{"name": n, "weight": w} for n, w in names_weights]

    def test_equal_weights(self):
        from eval.run_eval import compute_weighted_score
        criteria = self._make_criteria([("a", 1.0), ("b", 1.0), ("c", 1.0)])
        scores = {"a": 3, "b": 4, "c": 5}
        result = compute_weighted_score(scores, criteria)
        self.assertAlmostEqual(result, 4.0, places=2)

    def test_unequal_weights(self):
        from eval.run_eval import compute_weighted_score
        # clarity=5 (w=2), others=1 (w=1 each)
        criteria = self._make_criteria([("clarity", 2.0), ("completeness", 1.0)])
        scores = {"clarity": 5, "completeness": 1}
        # (5*2 + 1*1) / (2+1) = 11/3 ≈ 3.667
        result = compute_weighted_score(scores, criteria)
        self.assertAlmostEqual(result, 11/3, places=2)

    def test_all_fives(self):
        from eval.run_eval import compute_weighted_score
        criteria = self._make_criteria([("a", 2.0), ("b", 1.5), ("c", 1.0)])
        scores = {"a": 5, "b": 5, "c": 5}
        result = compute_weighted_score(scores, criteria)
        self.assertAlmostEqual(result, 5.0, places=2)

    def test_all_ones(self):
        from eval.run_eval import compute_weighted_score
        criteria = self._make_criteria([("x", 1.0), ("y", 1.0)])
        scores = {"x": 1, "y": 1}
        result = compute_weighted_score(scores, criteria)
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_missing_score_defaults_to_3(self):
        from eval.run_eval import compute_weighted_score
        criteria = self._make_criteria([("a", 1.0), ("b", 1.0)])
        scores = {"a": 5}  # "b" missing → defaults to 3
        result = compute_weighted_score(scores, criteria)
        self.assertAlmostEqual(result, 4.0, places=2)

    def test_real_spec_6_criteria(self):
        """Smoke test: full spec with scores of 4 → weighted avg should be 4."""
        from eval.run_eval import load_spec, compute_weighted_score
        criteria = load_spec(SPEC_PATH)
        scores = {c["name"]: 4 for c in criteria}
        result = compute_weighted_score(scores, criteria)
        self.assertAlmostEqual(result, 4.0, places=2)


# ── _parse_score ──────────────────────────────────────────────────────────────

class TestParseScore(unittest.TestCase):

    def test_standard_format(self):
        from eval.run_eval import _parse_score
        score, reasoning = _parse_score("Score: 4\nReasoning: Clear and actionable.")
        self.assertEqual(score, 4)
        self.assertIn("Clear", reasoning)

    def test_score_case_insensitive(self):
        from eval.run_eval import _parse_score
        score, _ = _parse_score("score: 3\nreasoning: Acceptable.")
        self.assertEqual(score, 3)

    def test_score_1_and_5(self):
        from eval.run_eval import _parse_score
        s1, _ = _parse_score("Score: 1\nReasoning: Very poor.")
        s5, _ = _parse_score("Score: 5\nReasoning: Excellent.")
        self.assertEqual(s1, 1)
        self.assertEqual(s5, 5)

    def test_defaults_to_3_on_parse_failure(self):
        from eval.run_eval import _parse_score
        score, _ = _parse_score("This is not a valid response at all.")
        self.assertEqual(score, 3)

    def test_reasoning_truncated_to_500(self):
        from eval.run_eval import _parse_score
        long_reason = "x" * 1000
        _, reasoning = _parse_score(f"Score: 3\nReasoning: {long_reason}")
        self.assertLessEqual(len(reasoning), 500)

    def test_multiline_reasoning_captured(self):
        from eval.run_eval import _parse_score
        text = "Score: 4\nReasoning: Good output.\nIt covers most patterns."
        score, reasoning = _parse_score(text)
        self.assertEqual(score, 4)
        self.assertGreater(len(reasoning), 10)


# ── _heuristic_score ──────────────────────────────────────────────────────────

class TestHeuristicScore(unittest.TestCase):

    def _score(self, name: str, doc: str) -> int:
        from eval.run_eval import _heuristic_score
        return _heuristic_score(doc, {"name": name})

    def test_returns_int_in_range(self):
        for criterion in ["clarity", "completeness", "actionability", "freshness", "anti_patterns"]:
            score = self._score(criterion, "some document text")
            self.assertIsInstance(score, int)
            self.assertGreaterEqual(score, 1)
            self.assertLessEqual(score, 5)

    def test_actionability_boosted_by_python_code(self):
        doc_with_code = "```python\nimport anthropic\ndef main(): pass\n```"
        doc_without = "This document talks about concepts in general terms."
        score_with = self._score("actionability", doc_with_code)
        score_without = self._score("actionability", doc_without)
        self.assertGreaterEqual(score_with, score_without)

    def test_anti_patterns_boosted_by_keywords(self):
        doc_with = "Avoid using global state. Never call this without arguments. Common mistake: forgetting to close."
        doc_without = "This function handles requests."
        score_with = self._score("anti_patterns", doc_with)
        score_without = self._score("anti_patterns", doc_without)
        self.assertGreaterEqual(score_with, score_without)

    def test_freshness_boosted_by_year(self):
        doc_with = "Updated for 2026 standards."
        doc_without = "This is a general guide."
        score_with = self._score("freshness", doc_with)
        score_without = self._score("freshness", doc_without)
        self.assertGreaterEqual(score_with, score_without)

    def test_unknown_criterion_returns_valid_score(self):
        score = self._score("unknown_criterion", "some text")
        self.assertGreaterEqual(score, 1)
        self.assertLessEqual(score, 5)


# ── Heuristic scoring bias fix ────────────────────────────────────────────────

class TestHeuristicScoringBiasFix(unittest.TestCase):
    """
    Regression tests for the scoring-bias fix (walkthrough finding):
    - clarity must be capped at 3 (not 4) even for keyword-rich documents
    - "example" (removed from clarity signals) must not inflate clarity score
    - freshness must be capped at 3 regardless of hit count
    """

    def _score(self, name: str, doc: str) -> int:
        from eval.run_eval import _heuristic_score
        return _heuristic_score(doc, {"name": name})

    def test_clarity_ceiling_is_3(self):
        # A document rich in formerly-positive clarity signals (e.g., "example")
        # should not exceed 3 with the new ceiling.
        rich_doc = (
            "For instance, consider this approach. "
            "Specifically, the pattern works like this: e.g. you call the function. "
            "For instance, another example can be found here."
        )
        score = self._score("clarity", rich_doc)
        self.assertLessEqual(score, 3, "clarity score must not exceed 3 (ceiling fix)")

    def test_clarity_example_keyword_no_longer_counts(self):
        # "example" was removed from clarity signals to prevent inflation.
        # A doc with only "example" should score the same as a doc with no signals.
        doc_with_only_example = "This is an example of a pattern. Another example follows."
        doc_no_signals = "This document discusses general concepts in a broad manner."
        score_example = self._score("clarity", doc_with_only_example)
        score_none = self._score("clarity", doc_no_signals)
        # Both should get the baseline score of 2 (no real clarity signals)
        self.assertEqual(score_example, score_none,
                         '"example" keyword must not raise clarity score above baseline')

    def test_freshness_ceiling_is_3(self):
        # A document with many freshness signals should still not exceed 3.
        rich_doc = "Updated for 2025 and 2026. Latest modern standards. Updated patterns."
        score = self._score("freshness", rich_doc)
        self.assertLessEqual(score, 3, "freshness score must not exceed 3 (ceiling fix)")

    def test_actionability_still_reaches_4(self):
        # Actionability ceiling remains 4 — not reduced.
        code_doc = "```python\nimport anthropic\ndef call(): pass\npip install anthropic\ncopy this\n```"
        score = self._score("actionability", code_doc)
        self.assertGreaterEqual(score, 3)
        self.assertLessEqual(score, 4)

    def test_anti_patterns_still_reaches_4(self):
        # anti_patterns ceiling remains 4 — not reduced.
        doc = "Don't use this pattern. Avoid calling without auth. Never ignore errors. Gotcha: missing close."
        score = self._score("anti_patterns", doc)
        self.assertLessEqual(score, 4)
        self.assertGreaterEqual(score, 3)


# ── evaluate() integration ────────────────────────────────────────────────────

class TestEvaluate(unittest.TestCase):

    def _mock_judge_response(self, score: int = 4, reasoning: str = "Good output."):
        return f"Score: {score}\nReasoning: {reasoning}"

    def test_evaluate_returns_report_dict(self):
        from eval.run_eval import evaluate

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Test Document\n\nSome content with ```python\nimport os\n```")
            tmp = Path(f.name)

        with tempfile.TemporaryDirectory() as out_dir:
            with patch("autoresearch.llm_client.chat", return_value=self._mock_judge_response()):
                report = evaluate(
                    input_path=tmp,
                    judge_model="ollama:llama3.2",
                    threshold=3.5,
                    output_path=Path(out_dir) / "report.json",
                    verbose=False,
                )

        self.assertIn("summary", report)
        self.assertIn("criteria", report)
        self.assertEqual(len(report["criteria"]), 6)

    def test_evaluate_writes_json_file(self):
        from eval.run_eval import evaluate

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Test\n\nContent here.")
            tmp = Path(f.name)

        with tempfile.TemporaryDirectory() as out_dir:
            out_path = Path(out_dir) / "out.json"
            with patch("autoresearch.llm_client.chat", return_value=self._mock_judge_response()):
                evaluate(
                    input_path=tmp,
                    output_path=out_path,
                    verbose=False,
                )
            self.assertTrue(out_path.exists())
            data = json.loads(out_path.read_text())
            self.assertIn("summary", data)

    def test_evaluate_pass_when_above_threshold(self):
        from eval.run_eval import evaluate

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# High quality output with concrete examples.")
            tmp = Path(f.name)

        with tempfile.TemporaryDirectory() as out_dir:
            with patch("autoresearch.llm_client.chat", return_value=self._mock_judge_response(score=5)):
                report = evaluate(
                    input_path=tmp,
                    threshold=3.5,
                    output_path=Path(out_dir) / "r.json",
                    verbose=False,
                )
        self.assertTrue(report["summary"]["passed"])

    def test_evaluate_fail_when_below_threshold(self):
        from eval.run_eval import evaluate

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Poor output.")
            tmp = Path(f.name)

        with tempfile.TemporaryDirectory() as out_dir:
            with patch("autoresearch.llm_client.chat", return_value=self._mock_judge_response(score=1)):
                report = evaluate(
                    input_path=tmp,
                    threshold=3.5,
                    output_path=Path(out_dir) / "r.json",
                    verbose=False,
                )
        self.assertFalse(report["summary"]["passed"])

    def test_evaluate_raises_on_missing_file(self):
        from eval.run_eval import evaluate
        with self.assertRaises(FileNotFoundError):
            evaluate(input_path="/nonexistent/path/doc.md", verbose=False)

    def test_evaluate_heuristic_fallback_when_llm_unavailable(self):
        """evaluate() should not crash even if LLM is unreachable."""
        from eval.run_eval import evaluate

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("```python\nimport anthropic\n# avoid these mistakes\n```")
            tmp = Path(f.name)

        with tempfile.TemporaryDirectory() as out_dir:
            # Simulate LLM failure
            with patch("autoresearch.llm_client.chat", side_effect=ConnectionError("no server")):
                report = evaluate(
                    input_path=tmp,
                    output_path=Path(out_dir) / "r.json",
                    verbose=False,
                )
        self.assertIn("summary", report)
        self.assertIn("weighted_avg", report["summary"])
        self.assertGreaterEqual(report["summary"]["weighted_avg"], 1.0)

    def test_summary_contains_required_fields(self):
        from eval.run_eval import evaluate

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Doc\n\nContent.")
            tmp = Path(f.name)

        with tempfile.TemporaryDirectory() as out_dir:
            with patch("autoresearch.llm_client.chat", return_value=self._mock_judge_response()):
                report = evaluate(
                    input_path=tmp,
                    judge_model="ollama:test",
                    threshold=3.0,
                    output_path=Path(out_dir) / "r.json",
                    verbose=False,
                )

        summary = report["summary"]
        for field in ["timestamp", "judge_model", "input_file", "threshold_used",
                       "weighted_avg", "passed", "criterion_scores"]:
            self.assertIn(field, summary, f"Missing field: {field}")

        self.assertEqual(summary["judge_model"], "ollama:test")
        self.assertEqual(summary["threshold_used"], 3.0)
        self.assertIsInstance(summary["criterion_scores"], dict)
        self.assertEqual(len(summary["criterion_scores"]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
