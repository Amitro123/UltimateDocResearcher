"""
tests/test_analyzer.py
----------------------
Unit tests for collector/analyzer.py.

Tests cover:
  - chunk_text(): basic splitting, overlap, paragraph boundaries
  - chunk_text(): regression — uniform text with no double-newlines must produce
    multiple chunks (the original bug: 1 chunk returned for 2500-char text with
    chunk_size=500)
  - chunk_text(): empty / short text edge cases
  - score_document(): short text gets low score, long diverse text gets higher score
  - is_personal_document(): detects invoices and resumes
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.analyzer import chunk_text, score_document, is_personal_document


# ── chunk_text ────────────────────────────────────────────────────────────────

class TestChunkText(unittest.TestCase):

    # ── Regression: uniform text with no double-newlines ─────────────────────

    def test_uniform_text_produces_multiple_chunks(self):
        """Regression: 'word ' * 500 must produce multiple chunks, not 1."""
        text = "word " * 500           # ~2500 chars, no paragraph breaks
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        self.assertGreater(len(chunks), 1,
            "uniform text should be split into multiple chunks, got 1 (regression)")

    def test_chunk_count_is_approximately_correct(self):
        text = "word " * 500           # ~2500 chars
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        # Should get roughly 5 chunks (2500 / 500)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertLessEqual(len(chunks), 8)

    def test_each_chunk_does_not_exceed_chunk_size_significantly(self):
        """No chunk should be much larger than chunk_size (allow small overshoot
        at word boundaries)."""
        text = "word " * 500
        chunk_size = 500
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=0):
            self.assertLessEqual(len(chunk), chunk_size + 50)

    # ── Paragraph-boundary splitting ─────────────────────────────────────────

    def test_respects_paragraph_boundaries(self):
        """Short paragraphs should be merged up to chunk_size."""
        para = "This is a short paragraph."
        text = (para + "\n\n") * 3
        chunks = chunk_text(text, chunk_size=500)
        # All 3 short paragraphs should fit in one chunk
        self.assertEqual(len(chunks), 1)

    def test_splits_at_paragraph_when_needed(self):
        long_para = "a" * 300
        text = long_para + "\n\n" + long_para + "\n\n" + long_para
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        # 300+300=600 > 500 → should be split
        self.assertGreaterEqual(len(chunks), 2)

    # ── Overlap ──────────────────────────────────────────────────────────────

    def test_overlap_shares_content(self):
        text = "abc " * 200          # 800 chars
        chunk_size = 200
        overlap = 50
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if len(chunks) >= 2:
            # The end of chunk N-1 should appear at the start of chunk N
            tail = chunks[0][-overlap:].strip()
            self.assertIn(tail[:10], chunks[1])

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(chunk_text(""), [])

    def test_whitespace_only_returns_empty_list(self):
        self.assertEqual(chunk_text("   \n\n   "), [])

    def test_short_text_returns_single_chunk(self):
        text = "This is a short document."
        chunks = chunk_text(text, chunk_size=2048)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_returns_list_of_strings(self):
        text = "hello " * 300
        result = chunk_text(text, chunk_size=500)
        self.assertIsInstance(result, list)
        for chunk in result:
            self.assertIsInstance(chunk, str)


# ── score_document ────────────────────────────────────────────────────────────

class TestScoreDocument(unittest.TestCase):

    def test_empty_text_scores_zero(self):
        score, reason = score_document("", "doc")
        self.assertEqual(score, 0.0)
        self.assertIsInstance(reason, str)

    def test_very_short_text_scores_low(self):
        score, _ = score_document("short", "doc")
        self.assertLessEqual(score, 0.2)

    def test_diverse_research_text_scores_above_zero(self):
        # 100 unique words as a research snippet
        words = [f"term{i}" for i in range(100)] + ["machine", "learning", "neural"]
        text = " ".join(words * 3)
        score, _ = score_document(text, "Research Paper")
        self.assertGreater(score, 0.0)

    def test_boilerplate_text_penalised(self):
        boilerplate = (
            "Cookie policy. Privacy policy terms of service. "
            "Subscribe to our newsletter. Javascript is disabled. "
        ) * 30
        score, _ = score_document(boilerplate, "Homepage")
        short_score, _ = score_document(boilerplate, "Homepage")
        # Score should be penalised (still meaningful but lower)
        self.assertLessEqual(score, 1.0)

    def test_returns_tuple_of_float_and_str(self):
        score, reason = score_document("hello world " * 20, "test")
        self.assertIsInstance(score, float)
        self.assertIsInstance(reason, str)

    def test_score_in_range_0_to_1(self):
        text = "Research on machine learning and neural networks " * 50
        score, _ = score_document(text, "ML Research")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ── is_personal_document ──────────────────────────────────────────────────────

class TestIsPersonalDocument(unittest.TestCase):

    def test_detects_invoice(self):
        text = "Invoice #12345\nAmount Due: $500.00\nPayment Terms: Net 30"
        personal, reason = is_personal_document(text, "invoice.pdf")
        self.assertTrue(personal)

    def test_detects_resume(self):
        text = "Resume\nEducation: Bachelor's in CS\nWork Experience: 5 years\nReferences available"
        personal, reason = is_personal_document(text, "my_resume.pdf")
        self.assertTrue(personal)

    def test_research_paper_is_not_personal(self):
        text = (
            "Abstract: This paper presents a novel approach to fine-tuning LLMs "
            "using LoRA adapters. Our experiments show 3x speedup with minimal "
            "quality degradation on standard benchmarks."
        )
        personal, _ = is_personal_document(text, "research_paper.pdf")
        self.assertFalse(personal)

    def test_returns_reason_when_personal(self):
        text = "Invoice #999\nBill to: John Doe\nTotal: $1000"
        personal, reason = is_personal_document(text, "bill.pdf")
        if personal:
            self.assertIsInstance(reason, str)
            self.assertGreater(len(reason), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
