"""
tests/test_generators.py
-------------------------
Unit tests for research_deliverables/generators.py.

Covers:
  - DeliverablePackage dataclass: construction, error dict
  - _render_template: renders known templates, falls back gracefully on unknown
  - _extract_sections: parses ## headings correctly
  - _extract_or_default: picks first matching key, returns default on miss
  - _llm_chat: returns LLM response, returns fallback string when LLM fails
  - generate_deliverables: end-to-end with mocked LLM and temp corpus
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_deliverables.generators import (
    DeliverablePackage,
    _render_template,
    _extract_sections,
    _extract_or_default,
    _llm_chat,
    _warn_if_sparse_sections,
    _try_json_extract,
    generate_deliverables,
)


# ── DeliverablePackage ────────────────────────────────────────────────────────

class TestDeliverablePackage(unittest.TestCase):

    def test_basic_construction(self):
        pkg = DeliverablePackage(
            run_id="test-run-001",
            topic="Claude tool use",
            research_type="code",
            output_dir=Path("/tmp/test"),
            files={"SUMMARY.md": Path("/tmp/test/SUMMARY.md")},
            metadata={"model": "gemini-2.5-flash-lite"},
        )
        self.assertEqual(pkg.run_id, "test-run-001")
        self.assertEqual(pkg.research_type, "code")
        self.assertEqual(pkg.errors, {})

    def test_errors_dict_defaults_empty(self):
        pkg = DeliverablePackage(
            run_id="x", topic="t", research_type="code",
            output_dir=Path("."), files={}, metadata={},
        )
        self.assertIsInstance(pkg.errors, dict)
        self.assertEqual(len(pkg.errors), 0)

    def test_errors_can_be_set(self):
        pkg = DeliverablePackage(
            run_id="x", topic="t", research_type="code",
            output_dir=Path("."), files={}, metadata={},
            errors={"SUMMARY.md": "LLM timed out"},
        )
        self.assertIn("SUMMARY.md", pkg.errors)


# ── _render_template ──────────────────────────────────────────────────────────

class TestRenderTemplate(unittest.TestCase):

    def test_renders_summary_template(self):
        result = _render_template("summary.jinja2", {
            "topic": "Claude tool use",
            "overview": "An overview.",
            "key_findings": "- Finding 1\n- Finding 2",
            "recommended_next_action": "Start building.",
        })
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        # Should contain the topic somewhere
        self.assertIn("Claude tool use", result)

    def test_renders_implementation_template(self):
        result = _render_template("implementation.jinja2", {
            "topic": "RAG patterns",
            "prerequisites": "- Python 3.11\n- anthropic SDK",
            "step_by_step_plan": "1. Install\n2. Configure\n3. Run",
            "key_apis": "anthropic.Anthropic()",
            "code_patterns": "```python\nchat()\n```",
            "common_pitfalls": "- Forget to close",
            "validation_checklist": "- Check output",
            "dependencies_tooling": "pip install anthropic",
        })
        self.assertIsInstance(result, str)
        self.assertIn("RAG patterns", result)

    def test_fallback_on_unknown_template(self):
        """Should not raise — returns key=value dump."""
        result = _render_template("nonexistent_template.jinja2", {
            "topic": "Test Topic",
            "content": "Some content here.",
        })
        self.assertIsInstance(result, str)
        self.assertIn("Test Topic", result)

    def test_fallback_includes_string_values(self):
        result = _render_template("nonexistent.jinja2", {
            "topic": "My Topic",
            "key_insight": "Important finding",
        })
        self.assertIn("Important finding", result)


# ── _extract_sections ─────────────────────────────────────────────────────────

class TestExtractSections(unittest.TestCase):

    def test_extracts_single_section(self):
        text = "## Overview\nThis is the overview text."
        sections = _extract_sections(text)
        self.assertIn("overview", sections)
        self.assertIn("overview text", sections["overview"])

    def test_extracts_multiple_sections(self):
        text = (
            "## Key Findings\n- Finding 1\n- Finding 2\n\n"
            "## Next Steps\nDo this first.\n"
        )
        sections = _extract_sections(text)
        self.assertIn("key_findings", sections)
        self.assertIn("next_steps", sections)

    def test_heading_normalized_to_snake_case(self):
        text = "## Key Findings\nsome content"
        sections = _extract_sections(text)
        self.assertIn("key_findings", sections)
        self.assertNotIn("Key Findings", sections)

    def test_empty_text_returns_empty_dict(self):
        sections = _extract_sections("")
        self.assertEqual(sections, {})

    def test_text_without_headings_returns_empty(self):
        sections = _extract_sections("Just some prose without any headings.")
        self.assertEqual(sections, {})

    def test_body_is_stripped(self):
        text = "## Overview\n\n   Content here   \n\n"
        sections = _extract_sections(text)
        self.assertEqual(sections["overview"], "Content here")


# ── _extract_or_default ───────────────────────────────────────────────────────

class TestExtractOrDefault(unittest.TestCase):

    def test_returns_first_matching_key(self):
        sections = {"overview": "The overview.", "summary": "The summary."}
        result = _extract_or_default(sections, "overview", "summary")
        self.assertEqual(result, "The overview.")

    def test_falls_through_to_second_key(self):
        sections = {"summary": "The summary."}
        result = _extract_or_default(sections, "overview", "summary")
        self.assertEqual(result, "The summary.")

    def test_returns_default_when_no_keys_match(self):
        sections = {"irrelevant": "content"}
        result = _extract_or_default(sections, "overview", "summary")
        self.assertEqual(result, "_No content generated._")

    def test_custom_default(self):
        sections = {}
        result = _extract_or_default(sections, "missing", default="CUSTOM DEFAULT")
        self.assertEqual(result, "CUSTOM DEFAULT")

    def test_normalizes_key_with_spaces(self):
        # Section key "key findings" → normalized "key_findings"
        sections = {"key_findings": "found things"}
        result = _extract_or_default(sections, "key findings")
        self.assertEqual(result, "found things")


# ── _llm_chat ─────────────────────────────────────────────────────────────────

class TestLlmChat(unittest.TestCase):

    def test_returns_llm_response_on_success(self):
        with patch("autoresearch.llm_client.chat", return_value="LLM answer"):
            result = _llm_chat(
                system="You are helpful.",
                user="What is RAG?",
                model="ollama:llama3.2",
            )
        self.assertEqual(result, "LLM answer")

    def test_returns_fallback_string_on_exception(self):
        with patch("autoresearch.llm_client.chat", side_effect=ConnectionError("refused")):
            result = _llm_chat(
                system="You are helpful.",
                user="What is RAG?",
                model="ollama:llama3.2",
            )
        self.assertIn("LLM unavailable", result)
        self.assertIsInstance(result, str)


# ── generate_deliverables ─────────────────────────────────────────────────────

class TestGenerateDeliverables(unittest.TestCase):

    _LLM_RESPONSE = """\
## Overview
This is an overview of the research topic.

## Key Findings
- Finding 1
- Finding 2

## Recommended Next Action
Start implementing.
"""

    def _make_corpus(self, tmp_dir: Path) -> Path:
        corpus = tmp_dir / "corpus.txt"
        corpus.write_text(
            "This is a test corpus about Claude tool use and RAG patterns. " * 50,
            encoding="utf-8",
        )
        return corpus

    def test_returns_deliverable_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self._make_corpus(tmp)
            out_dir = tmp / "results"

            with patch("autoresearch.llm_client.chat", return_value=self._LLM_RESPONSE):
                pkg = generate_deliverables(
                    topic="Claude tool use patterns",
                    corpus_path=corpus,
                    output_dir=str(out_dir),
                    model="ollama:llama3.2",
                )

        self.assertIsInstance(pkg, DeliverablePackage)
        self.assertEqual(pkg.topic, "Claude tool use patterns")

    def test_creates_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self._make_corpus(tmp)
            out_dir = tmp / "results"

            with patch("autoresearch.llm_client.chat", return_value=self._LLM_RESPONSE):
                pkg = generate_deliverables(
                    topic="RAG implementation patterns",
                    corpus_path=corpus,
                    output_dir=str(out_dir),
                    model="ollama:llama3.2",
                    include_code=False,
                )

            # Assert inside the context manager so the temp dir still exists
            for name, path in pkg.files.items():
                self.assertTrue(path.exists(), f"{name} was not written to disk")
                self.assertGreater(path.stat().st_size, 0, f"{name} is empty")

    def test_metadata_contains_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self._make_corpus(tmp)

            with patch("autoresearch.llm_client.chat", return_value=self._LLM_RESPONSE):
                pkg = generate_deliverables(
                    topic="Claude agent SDK",
                    corpus_path=corpus,
                    model="ollama:llama3.2",
                    include_code=False,
                )

        for field in ["run_id", "topic", "research_type", "timestamp", "model"]:
            self.assertIn(field, pkg.metadata, f"Missing metadata field: {field}")

    def test_research_type_set_correctly_for_code_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self._make_corpus(tmp)

            with patch("autoresearch.llm_client.chat", return_value=self._LLM_RESPONSE):
                pkg = generate_deliverables(
                    topic="Python SDK integration patterns",
                    corpus_path=corpus,
                    model="ollama:llama3.2",
                    include_code=False,
                )

        self.assertEqual(pkg.research_type, "code")

    def test_research_type_set_correctly_for_arch_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self._make_corpus(tmp)

            with patch("autoresearch.llm_client.chat", return_value=self._LLM_RESPONSE):
                pkg = generate_deliverables(
                    topic="Architecture of a streaming data pipeline",
                    corpus_path=corpus,
                    model="ollama:llama3.2",
                    include_code=False,
                )

        self.assertEqual(pkg.research_type, "arch")

    def test_run_id_auto_generated_when_not_provided(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self._make_corpus(tmp)

            with patch("autoresearch.llm_client.chat", return_value=self._LLM_RESPONSE):
                pkg = generate_deliverables(
                    topic="RAG patterns",
                    corpus_path=corpus,
                    model="ollama:llama3.2",
                    include_code=False,
                )

        self.assertIsInstance(pkg.run_id, str)
        self.assertGreater(len(pkg.run_id), 0)

    def test_custom_run_id_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self._make_corpus(tmp)

            with patch("autoresearch.llm_client.chat", return_value=self._LLM_RESPONSE):
                pkg = generate_deliverables(
                    topic="RAG patterns",
                    corpus_path=corpus,
                    run_id="my-custom-run",
                    model="ollama:llama3.2",
                    include_code=False,
                )

        self.assertEqual(pkg.run_id, "my-custom-run")

    def test_raises_on_missing_corpus(self):
        with self.assertRaises((FileNotFoundError, Exception)):
            generate_deliverables(
                topic="test",
                corpus_path=Path("/nonexistent/corpus.txt"),
                model="ollama:llama3.2",
            )

    def test_llm_failure_recorded_in_errors_not_raised(self):
        """LLM errors per-deliverable should be caught, not crash the whole run."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self._make_corpus(tmp)

            # LLM always fails
            with patch("autoresearch.llm_client.chat",
                       side_effect=ConnectionError("no server")):
                pkg = generate_deliverables(
                    topic="Claude tool use",
                    corpus_path=corpus,
                    model="ollama:llama3.2",
                    include_code=False,
                )

        # Package is always returned — errors must not crash the caller
        self.assertIsInstance(pkg, DeliverablePackage)
        # Errors must be recorded in pkg.errors (not silently swallowed)
        self.assertGreater(len(pkg.errors), 0,
                           "LLM failures must land in pkg.errors, not be swallowed")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ── _llm_chat raise_on_error ──────────────────────────────────────────────────

class TestLlmChatRaiseOnError(unittest.TestCase):

    def test_raises_when_raise_on_error_true(self):
        """raise_on_error=True must re-raise instead of returning placeholder."""
        with patch("autoresearch.llm_client.chat", side_effect=ConnectionError("down")):
            with self.assertRaises(ConnectionError):
                _llm_chat(
                    system="sys",
                    user="user",
                    model="ollama:llama3.2",
                    raise_on_error=True,
                )

    def test_returns_placeholder_by_default(self):
        """Default behaviour (raise_on_error=False) must still return a string."""
        with patch("autoresearch.llm_client.chat", side_effect=ConnectionError("down")):
            result = _llm_chat(
                system="sys",
                user="user",
                model="ollama:llama3.2",
            )
        self.assertIn("LLM unavailable", result)
        self.assertIsInstance(result, str)


# ── _warn_if_sparse_sections ──────────────────────────────────────────────────

class TestWarnIfSparseSections(unittest.TestCase):

    def test_no_warning_when_all_sections_found(self):
        sections = {"overview": "x", "key_findings": "y", "recommended_next_action": "z"}
        import io
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            _warn_if_sparse_sections(
                "TEST.md", sections,
                ["Overview", "Key Findings", "Recommended Next Action"],
            )
            self.assertEqual(mock_err.getvalue(), "")

    def test_warning_when_below_50_pct(self):
        """Only 1 out of 3 expected sections present → must warn."""
        sections = {"overview": "text"}
        import io
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            _warn_if_sparse_sections(
                "SUMMARY.md", sections,
                ["Overview", "Key Findings", "Recommended Next Action"],
            )
            self.assertIn("33%", mock_err.getvalue())

    def test_no_warning_on_empty_expected_keys(self):
        """Empty expected list must not raise or warn."""
        import io
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            _warn_if_sparse_sections("TEST.md", {"a": "b"}, [])
            self.assertEqual(mock_err.getvalue(), "")

    def test_positional_aliases_not_counted_as_content(self):
        """_section_N keys are positional aliases and must not count toward coverage."""
        sections = {"_section_0": "x", "_section_1": "y"}
        import io
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            _warn_if_sparse_sections(
                "TEST.md", sections,
                ["Overview", "Key Findings"],
            )
            self.assertIn("0%", mock_err.getvalue())


# ── _try_json_extract ─────────────────────────────────────────────────────────

class TestTryJsonExtract(unittest.TestCase):

    def test_extracts_from_fenced_block(self):
        text = '```json\n{"overview": "hello", "key_findings": "world"}\n```'
        result = _try_json_extract(text, ["overview", "key_findings"])
        self.assertIsNotNone(result)
        self.assertEqual(result["overview"], "hello")

    def test_extracts_bare_json_object(self):
        text = '{"overview": "hello", "key_findings": "world"}'
        result = _try_json_extract(text, ["overview", "key_findings"])
        self.assertIsNotNone(result)

    def test_returns_none_on_missing_keys(self):
        text = '{"overview": "hello"}'
        result = _try_json_extract(text, ["overview", "key_findings"])
        self.assertIsNone(result)

    def test_returns_none_on_invalid_json(self):
        result = _try_json_extract("not json at all", ["overview"])
        self.assertIsNone(result)

    def test_returns_none_on_empty_text(self):
        result = _try_json_extract("", ["overview"])
        self.assertIsNone(result)

    def test_coerces_values_to_strings(self):
        text = '{"score": 42, "label": true}'
        result = _try_json_extract(text, ["score", "label"])
        self.assertIsNotNone(result)
        self.assertEqual(result["score"], "42")
        self.assertEqual(result["label"], "True")

    def test_handles_prose_before_json(self):
        text = 'Here is the result:\n{"overview": "content", "key_findings": "data"}'
        result = _try_json_extract(text, ["overview", "key_findings"])
        self.assertIsNotNone(result)

