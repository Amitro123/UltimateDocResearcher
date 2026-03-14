"""
tests/test_code_suggester.py
-----------------------------
Unit tests for autoresearch/code_suggester.py.

Tests cover:
  - Topic auto-detection from corpus + program.md
  - Corpus sampling strategy (spread across text, respects max_chars)
  - Heuristic fallback suggestions (no API key)
  - Full generate_suggestions pipeline with temp files
  - Output file creation and Markdown structure
  - Domain keyword detection
"""
import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoresearch.code_suggester import (
    _detect_topic,
    _sample_corpus,
    _heuristic_suggestions,
    _wrap_in_report,
    generate_suggestions,
)


# ── Topic detection ───────────────────────────────────────────────────────────

class TestDetectTopic(unittest.TestCase):

    def test_reads_topic_from_program_md_hash_header(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Claude Tool Use Patterns\n\nSome content here.\n")
            fname = f.name
        topic = _detect_topic("some corpus text", Path(fname))
        self.assertIn("Claude Tool Use Patterns", topic)

    def test_reads_topic_from_program_md_topic_label(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Topic: LoRA Fine-Tuning\n\nResearch details.\n")
            fname = f.name
        topic = _detect_topic("some corpus text", Path(fname))
        self.assertIn("LoRA Fine-Tuning", topic)

    def test_falls_back_to_corpus_bigrams_when_no_program_md(self):
        corpus = "anthropic tool use claude tool use anthropic sdk anthropic sdk anthropic sdk"
        topic = _detect_topic(corpus, program_md_path=None)
        # Should return some bigram-based topic
        self.assertIsInstance(topic, str)
        self.assertGreater(len(topic), 0)

    def test_returns_fallback_string_on_empty_corpus_and_no_file(self):
        topic = _detect_topic("", program_md_path=None)
        self.assertIsInstance(topic, str)


# ── Corpus sampling ───────────────────────────────────────────────────────────

class TestSampleCorpus(unittest.TestCase):

    def test_short_corpus_returned_unchanged(self):
        text = "short text"
        self.assertEqual(_sample_corpus(text, max_chars=1000), text)

    def test_long_corpus_truncated_to_max(self):
        text = "word " * 10000  # ~50k chars
        result = _sample_corpus(text, max_chars=3000)
        self.assertLessEqual(len(result), 4000)  # some overhead from separators

    def test_long_corpus_contains_beginning_and_end(self):
        text = "START " + ("middle " * 1000) + "END"
        result = _sample_corpus(text, max_chars=500)
        self.assertIn("START", result)
        self.assertIn("END", result)


# ── Heuristic fallback ────────────────────────────────────────────────────────

class TestHeuristicSuggestions(unittest.TestCase):

    def test_returns_markdown_string(self):
        result = _heuristic_suggestions("some corpus text", "AI engineering")
        self.assertIsInstance(result, str)
        self.assertIn("```", result)  # should have code block

    def test_detects_anthropic_domain(self):
        corpus = "anthropic sdk tool_use claude messages create tool_use block"
        result = _heuristic_suggestions(corpus, "")
        self.assertIn("Anthropic", result)

    def test_detects_lora_domain(self):
        corpus = "lora fine-tuning rank decomposition matrices transformers peft"
        result = _heuristic_suggestions(corpus, "")
        self.assertIn("LoRA", result)

    def test_fallback_includes_rerun_instructions(self):
        result = _heuristic_suggestions("some text", "any topic")
        self.assertIn("ANTHROPIC_API_KEY", result)


# ── Report wrapper ────────────────────────────────────────────────────────────

class TestWrapInReport(unittest.TestCase):

    def test_header_contains_topic(self):
        result = _wrap_in_report("## Snippet\n```python\npass\n```", "Claude SDK", {})
        self.assertIn("Claude SDK", result)

    def test_footer_contains_how_to_use(self):
        result = _wrap_in_report("content", "topic", {})
        self.assertIn("How to use these suggestions", result)

    def test_stats_rendered_in_header(self):
        stats = {"chunks": 42, "chars": 10000, "timestamp": "2026-01-01"}
        result = _wrap_in_report("content", "topic", stats)
        self.assertIn("42", result)
        self.assertIn("10,000", result)


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestGenerateSuggestions(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.corpus_path = Path(self.tmp) / "corpus.txt"
        self.output_path = Path(self.tmp) / "suggestions.md"

    def _write_corpus(self, text: str):
        self.corpus_path.write_text(text, encoding="utf-8")

    def test_output_file_is_created(self):
        self._write_corpus(
            "The Anthropic SDK supports tool use via the tools parameter.\n\n"
            "Tool definitions require name, description, and input_schema.\n\n"
            "Responses with tool_use content blocks must be handled by the caller."
        )
        generate_suggestions(
            corpus_path=self.corpus_path,
            topic="Claude tool use",
            model="gpt-4o-mini",  # heuristic fallback
            output_path=self.output_path,
        )
        self.assertTrue(self.output_path.exists())

    def test_output_is_valid_markdown(self):
        self._write_corpus("LoRA fine-tuning reduces parameters via rank decomposition.\n\n" * 5)
        result = generate_suggestions(
            corpus_path=self.corpus_path,
            topic="LoRA",
            model="gpt-4o-mini",
            output_path=self.output_path,
        )
        self.assertIn("# ", result)     # has at least one header
        self.assertIn("```", result)    # has at least one code block

    def test_topic_appears_in_output(self):
        self._write_corpus("some content about testing\n\n" * 3)
        result = generate_suggestions(
            corpus_path=self.corpus_path,
            topic="My Research Topic",
            model="gpt-4o-mini",
            output_path=self.output_path,
        )
        self.assertIn("My Research Topic", result)

    def test_raises_on_missing_corpus(self):
        with self.assertRaises(FileNotFoundError):
            generate_suggestions(
                corpus_path=Path(self.tmp) / "nonexistent.txt",
                output_path=self.output_path,
            )

    def test_output_dir_created_if_missing(self):
        nested_output = Path(self.tmp) / "deep" / "nested" / "suggestions.md"
        self._write_corpus("content\n\n" * 3)
        generate_suggestions(
            corpus_path=self.corpus_path,
            topic="test",
            model="gpt-4o-mini",
            output_path=nested_output,
        )
        self.assertTrue(nested_output.exists())

    def test_n_suggestions_param_accepted(self):
        self._write_corpus("content\n\n" * 5)
        # Should not raise regardless of n_suggestions value
        generate_suggestions(
            corpus_path=self.corpus_path,
            topic="test",
            model="gpt-4o-mini",
            output_path=self.output_path,
            n_suggestions=10,
        )
        self.assertTrue(self.output_path.exists())

    def test_auto_topic_detection_when_topic_empty(self):
        corpus = "anthropic claude sdk messages tool_use " * 50
        self._write_corpus(corpus)
        result = generate_suggestions(
            corpus_path=self.corpus_path,
            topic="",  # should auto-detect
            model="gpt-4o-mini",
            output_path=self.output_path,
        )
        # Topic should have been detected (non-empty) and appear in output
        self.assertGreater(len(result), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
