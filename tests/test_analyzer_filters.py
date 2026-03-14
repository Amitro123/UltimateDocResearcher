"""
tests/test_analyzer_filters.py
-------------------------------
Tests for the personal-document and language filters added to analyzer.py.

Covers:
  - is_personal_document: invoice, receipt, contract, CV, medical record patterns
  - is_personal_document: title-level detection
  - is_non_research_language: Hebrew, Arabic, CJK scripts
  - score_document: returns 0.0 for personal/non-research docs
  - score_document: normal docs still score > 0
  - analyze_corpus: personal docs excluded, reported in filtered_docs
  - analyze_corpus: non-research language excluded
  - analyze_corpus: clean research docs pass through
  - _warn_if_personal_folder: triggers on Downloads / Documents paths
"""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

# Stub optional heavy deps before any collector import
for _mod in ("aiohttp", "aiofiles"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.analyzer import (
    is_personal_document,
    is_non_research_language,
    score_document,
    analyze_corpus,
    dominant_script,
)

# Load _warn_if_personal_folder directly (aiohttp already stubbed above)
from collector.ultimate_collector import _warn_if_personal_folder


# ── is_personal_document ─────────────────────────────────────────────────────

class TestIsPersonalDocument(unittest.TestCase):

    def _assert_personal(self, text="", title=""):
        flag, reason = is_personal_document(text, title)
        self.assertTrue(flag, f"Expected personal, got clean. Reason: {reason!r}")
        self.assertTrue(reason)

    def _assert_clean(self, text="", title=""):
        flag, reason = is_personal_document(text, title)
        self.assertFalse(flag, f"Expected clean, got personal. Reason: {reason!r}")

    def test_invoice_number_in_body(self):
        self._assert_personal(text="Invoice number: INV-20240101\nTotal to pay: $150.00")

    def test_receipt_in_body(self):
        self._assert_personal(text="Receipt number 84729\nPayment received: $45.00")

    def test_order_number_in_body(self):
        self._assert_personal(text="Order number: 12345\nShipped to: 10 Main St")

    def test_bank_account_in_body(self):
        self._assert_personal(text="Bank account number: 1234567890\nIBAN: GB29NWBK")

    def test_contract_signature_in_body(self):
        self._assert_personal(
            text="The employee hereby agrees to the terms and conditions signed below."
        )

    def test_cv_pattern(self):
        self._assert_personal(
            text="Curriculum Vitae\nWork Experience\nEducation\nSkills"
        )

    def test_invoice_title(self):
        self._assert_personal(title="invoice_1234_for_order_5678")

    def test_offer_letter_title(self):
        self._assert_personal(title="Offer Letter - signed")

    def test_resume_title(self):
        self._assert_personal(title="Amit-Rosen-Resume-2026")

    def test_research_paper_not_flagged(self):
        self._assert_clean(
            text="This paper presents a novel approach to fine-tuning language models "
                 "using LoRA adapters on a custom dataset.",
            title="LoRA-finetuning-survey"
        )

    def test_claude_skills_guide_not_flagged(self):
        self._assert_clean(
            text="Building skills for Claude requires a well-structured SKILL.md file "
                 "with a clear description that triggers the skill appropriately.",
            title="The-Complete-Guide-to-Building-Skill-for-Claude"
        )

    def test_code_snippet_not_flagged(self):
        self._assert_clean(
            text="```python\nimport anthropic\nclient = anthropic.Anthropic()\n```"
        )


# ── is_non_research_language ──────────────────────────────────────────────────

class TestIsNonResearchLanguage(unittest.TestCase):

    def test_hebrew_heavy_text_flagged(self):
        hebrew = "שלום עולם " * 50  # well above 20% threshold
        flag, reason = is_non_research_language(hebrew)
        self.assertTrue(flag)
        self.assertIn("hebrew", reason.lower())

    def test_arabic_heavy_text_flagged(self):
        arabic = "مرحبا بالعالم " * 50
        flag, reason = is_non_research_language(arabic)
        self.assertTrue(flag)
        self.assertIn("arabic", reason.lower())

    def test_english_text_not_flagged(self):
        english = "This is a research paper about machine learning and fine-tuning." * 10
        flag, reason = is_non_research_language(english)
        self.assertFalse(flag)

    def test_mixed_mostly_english_not_flagged(self):
        # Some Hebrew characters but overwhelmingly English
        mixed = ("This paper discusses AI. " * 20) + "שלום"
        flag, _ = is_non_research_language(mixed)
        self.assertFalse(flag)

    def test_chinese_heavy_text_flagged(self):
        chinese = "你好世界这是一段中文文本" * 30
        flag, reason = is_non_research_language(chinese)
        self.assertTrue(flag)


# ── dominant_script ───────────────────────────────────────────────────────────

class TestDominantScript(unittest.TestCase):

    def test_latin_text(self):
        script, frac = dominant_script("Hello world, this is English text.")
        self.assertEqual(script, "latin")

    def test_hebrew_text(self):
        script, frac = dominant_script("שלום עולם " * 20)
        self.assertEqual(script, "hebrew")
        self.assertGreater(frac, 0.2)

    def test_arabic_text(self):
        script, frac = dominant_script("مرحبا " * 20)
        self.assertEqual(script, "arabic")


# ── score_document ────────────────────────────────────────────────────────────

class TestScoreDocument(unittest.TestCase):

    def test_invoice_scores_zero(self):
        text = "Invoice number: INV-001\nTotal to pay: $200.00\nPayment received."
        score, reason = score_document(text, "invoice_001")
        self.assertEqual(score, 0.0)
        self.assertTrue(reason)

    def test_hebrew_invoice_scores_zero(self):
        # Hebrew invoice-style text
        hebrew = "חשבונית מס" + " " + "שלום עולם " * 30
        score, reason = score_document(hebrew, "hebrew_invoice")
        self.assertEqual(score, 0.0)

    def test_research_doc_scores_above_threshold(self):
        # Use naturally varied text (no repetition) to preserve a good TTR
        text = (
            "Building skills for Claude requires a well-structured SKILL.md file. "
            "The description field determines when Claude invokes the skill. "
            "Use specific trigger phrases that users would naturally type. "
            "Always test your skill against representative prompts before deploying. "
            "A good description mentions the exact output format and use cases. "
            "Fine-tuning with LoRA adapters enables parameter-efficient training. "
            "The research corpus should be topic-specific and carefully curated. "
            "Clarity is the most important criterion for evaluating research outputs. "
            "Actionable suggestions contain working imports and copy-paste code. "
            "Anti-pattern sections explicitly document what developers should avoid. "
            "Freshness ensures recommendations reflect 2025 and 2026 best practices. "
            "Completeness means covering ninety percent of documented corpus patterns. "
            "The eval framework assigns weighted scores across five distinct criteria. "
            "Ollama provides a free local LLM backend requiring no API key or network. "
        )
        score, reason = score_document(text, "claude-skills-guide")
        self.assertGreater(score, 0.25)
        self.assertEqual(reason, "")

    def test_empty_doc_scores_zero(self):
        score, reason = score_document("")
        self.assertEqual(score, 0.0)

    def test_short_doc_low_score(self):
        score, reason = score_document("Short text.")
        self.assertLess(score, 0.25)

    def test_returns_tuple(self):
        result = score_document("Some text here for testing purposes.")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ── analyze_corpus integration ────────────────────────────────────────────────

_RESEARCH_DOC = """\
=== claude-skills-guide [pdf] ===
Building skills for Claude requires a well-structured SKILL.md file containing
a description that determines when Claude will invoke the skill automatically.

The description field is the most critical part of any skill definition.
It should begin with an action verb, list the exact output format produced,
and include three to five trigger phrases that users would naturally say.

When testing your skill, run it against representative prompts to verify
it fires on the intended requests and stays silent on unrelated ones.
Testing iteration is faster than trying to get the description perfect upfront.

Fine-tuning language models with LoRA adapters allows parameter-efficient
training by injecting trainable rank-decomposition matrices into transformer
layers without modifying the pretrained weights directly.

The research corpus quality directly impacts the quality of generated
training pairs. Include diverse sources covering edge cases, anti-patterns,
and concrete worked examples rather than abstract conceptual descriptions.

A well-defined eval spec uses weighted criteria to score research outputs.
Clarity carries the highest weight because ambiguous instructions cause
misuse, while freshness ensures the output reflects current best practices.

Actionable outputs contain working imports, real function signatures,
and copy-paste ready code rather than pseudocode or placeholder values.
Anti-pattern sections prevent misuse by explicitly documenting what not to do.

The autoresearch pipeline collects documents, analyzes quality, generates
Q&A training pairs, and optionally fine-tunes a model on the resulting dataset.
Each iteration produces an eval report showing whether quality improved.
"""

_INVOICE_DOC = """\
=== invoice_12345_for_order_99999 [pdf] ===
Invoice number: INV-12345
Order number: 99999
Total to pay: $199.99
Payment method: Credit card
Shipped to: 10 Main Street
"""

_HEBREW_DOC = """\
=== hebrew_document [pdf] ===
""" + "שלום עולם זוהי מסמך בעברית " * 60


class TestAnalyzeCorpus(unittest.TestCase):

    def _write_corpus(self, tmp_dir: Path, *docs: str) -> Path:
        corpus = Path(tmp_dir) / "all_docs.txt"
        corpus.write_text("\n<DOC_SEP>\n".join(docs), encoding="utf-8")
        return corpus

    def test_research_doc_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = self._write_corpus(tmp, _RESEARCH_DOC)
            report = analyze_corpus(corpus, tmp, verbose=False)
            self.assertGreater(report["docs_passing_filter"], 0)
            cleaned = Path(tmp) / "all_docs_cleaned.txt"
            self.assertGreater(cleaned.stat().st_size, 0)

    def test_invoice_filtered_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = self._write_corpus(tmp, _RESEARCH_DOC, _INVOICE_DOC)
            report = analyze_corpus(corpus, tmp, verbose=False)
            self.assertGreater(report["docs_filtered_personal"], 0)
            # Invoice should appear in filtered_docs list
            titles = [d["title"] for d in report["filtered_docs"]]
            self.assertTrue(any("invoice" in t.lower() for t in titles))

    def test_hebrew_doc_filtered_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = self._write_corpus(tmp, _RESEARCH_DOC, _HEBREW_DOC)
            report = analyze_corpus(corpus, tmp, verbose=False)
            self.assertGreater(report["docs_filtered_personal"], 0)

    def test_only_junk_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = self._write_corpus(tmp, _INVOICE_DOC, _HEBREW_DOC)
            captured = StringIO()
            with patch("sys.stdout", captured):
                analyze_corpus(corpus, tmp, verbose=True)
            output = captured.getvalue()
            self.assertIn("WARNING", output)

    def test_report_json_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = self._write_corpus(tmp, _RESEARCH_DOC)
            analyze_corpus(corpus, tmp, verbose=False)
            report_path = Path(tmp) / "corpus_report.json"
            self.assertTrue(report_path.exists())
            data = json.loads(report_path.read_text())
            self.assertIn("filtered_docs", data)
            self.assertIn("docs_filtered_personal", data)

    def test_filtered_docs_have_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = self._write_corpus(tmp, _RESEARCH_DOC, _INVOICE_DOC)
            report = analyze_corpus(corpus, tmp, verbose=False)
            for fd in report["filtered_docs"]:
                self.assertIn("reason", fd)
                self.assertTrue(fd["reason"])


# ── _warn_if_personal_folder ──────────────────────────────────────────────────

class TestWarnIfPersonalFolder(unittest.TestCase):

    def _capture_warn(self, path_str: str) -> str:
        captured = StringIO()
        with patch("sys.stdout", captured):
            _warn_if_personal_folder(Path(path_str))
        return captured.getvalue()

    def test_downloads_triggers_warning(self):
        out = self._capture_warn("/home/user/Downloads")
        self.assertIn("WARNING", out)
        self.assertIn("papers/", out)

    def test_documents_triggers_warning(self):
        out = self._capture_warn("C:/Users/Dana/Documents")
        self.assertIn("WARNING", out)

    def test_desktop_triggers_warning(self):
        out = self._capture_warn("/home/user/Desktop")
        self.assertIn("WARNING", out)

    def test_papers_folder_no_warning(self):
        out = self._capture_warn("/home/user/projects/papers")
        self.assertEqual(out, "")

    def test_research_folder_no_warning(self):
        out = self._capture_warn("/home/user/research/pdfs")
        self.assertEqual(out, "")

    def test_nested_downloads_triggers_warning(self):
        # e.g. Downloads/subfolder
        out = self._capture_warn("/home/user/Downloads/research_pdfs")
        self.assertIn("WARNING", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
