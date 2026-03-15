"""
tests/test_external_focus.py
-----------------------------
Tests for:
  - is_internal_doc()       — correct source type classification
  - analyze_corpus()        — writes external_docs.txt, correct fraction in report
  - _sample_corpus_weighted() — respects 70/30 external/internal split
  - eval_spec.yaml          — external_novelty criterion present and well-formed
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── is_internal_doc ────────────────────────────────────────────────────────────

from collector.analyzer import is_internal_doc


class TestIsInternalDoc:
    def test_pdf_source_always_external(self):
        assert not is_internal_doc("UltimateDocResearcher code review", source="pdf")

    def test_web_source_always_external(self):
        assert not is_internal_doc("E2E_GUIDE best practices", source="web")

    def test_github_source_always_external(self):
        assert not is_internal_doc("UltimateDocResearcher AGENTS.md", source="github")

    def test_reddit_source_always_external(self):
        assert not is_internal_doc("r/LocalLLaMA post", source="reddit")

    def test_local_internal_title_detected(self):
        assert is_internal_doc("CODE_REVIEW analysis", source="local")

    def test_local_udr_title_detected(self):
        assert is_internal_doc("UltimateDocResearcher walkthrough", source="local")

    def test_local_agents_md_detected(self):
        assert is_internal_doc("agents.md phase plans", source="local")

    def test_local_external_paper_not_flagged(self):
        assert not is_internal_doc("A Practical Guide to Evaluating LLMs", source="local")

    def test_local_external_random_doc_not_flagged(self):
        assert not is_internal_doc("Retrieval Augmented Generation Survey 2024", source="local")

    def test_case_insensitive(self):
        assert is_internal_doc("ULTIMATEDOCRESEARCHER reliability", source="local")

    def test_custom_cfg_internal_patterns(self):
        cfg = {"sources": {"internal_title_patterns": ["myproject"], "always_external_sources": ["pdf"]}}
        assert is_internal_doc("myproject setup guide", source="local", cfg=cfg)
        assert not is_internal_doc("myproject setup guide", source="pdf", cfg=cfg)


# ── analyze_corpus external tagging ───────────────────────────────────────────

from collector.analyzer import analyze_corpus


_LOREM = (
    "Retrieval augmented generation combines dense retrieval with neural text generation "
    "to ground language model outputs in verified documents. The retriever encodes queries "
    "into vector embeddings and performs approximate nearest-neighbor search over an index. "
    "Top-k passages are prepended to the prompt before the generative model produces an answer. "
    "This approach significantly reduces hallucination compared to closed-book inference. "
    "Evaluation typically measures exact match, F1, and faithfulness against source passages. "
    "Recent work explores iterative retrieval, where the model generates intermediate queries "
    "to fill knowledge gaps discovered during decoding. Passage reranking with cross-encoders "
    "further improves precision before the final answer step. Practical deployments cache "
    "document embeddings and refresh indices incrementally as corpora update. "
)

def _rich_text(seed: str = "") -> str:
    """Return varied research-quality text that passes the quality filter."""
    return (seed + " " + _LOREM * 3).strip()


def _make_all_docs(docs: list[dict]) -> str:
    """Build an all_docs.txt fixture from a list of {title, source, text} dicts."""
    parts = []
    for d in docs:
        parts.append(f"=== {d['title']} [{d['source']}] ===\n{d['text']}")
    return "<DOC_SEP>".join(parts)


class TestAnalyzeCorpusExternalTagging:
    def _run(self, docs: list[dict]) -> tuple[dict, Path]:
        """Run analyze_corpus on fixture docs and return (report, output_dir)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            all_docs_path = tmp / "all_docs.txt"
            all_docs_path.write_text(_make_all_docs(docs), encoding="utf-8")
            report = analyze_corpus(all_docs_path, output_dir=tmp, verbose=False)
            yield report, tmp

    # Use a low quality_threshold in these tests — we're testing source tagging,
    # not the quality filter (which is covered by test_analyzer.py).
    _THRESHOLD = 0.05

    def test_external_docs_file_created_when_external_exists(self):
        docs = [
            {"title": "RAG survey paper", "source": "pdf",
             "text": _rich_text("EXTERNAL_MARKER")},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "all_docs.txt").write_text(_make_all_docs(docs), encoding="utf-8")
            report = analyze_corpus(tmp / "all_docs.txt", output_dir=tmp,
                                    quality_threshold=self._THRESHOLD, verbose=False)
            assert (tmp / "external_docs.txt").exists(), "external_docs.txt should be created"
            assert report["external_chunks"] > 0
            assert report["internal_chunks"] == 0
            assert report["external_fraction"] == 1.0

    def test_internal_docs_not_in_external_file(self):
        docs = [
            {"title": "CODE_REVIEW analysis", "source": "local",
             "text": _rich_text("INTERNAL_MARKER")},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "all_docs.txt").write_text(_make_all_docs(docs), encoding="utf-8")
            report = analyze_corpus(tmp / "all_docs.txt", output_dir=tmp,
                                    quality_threshold=self._THRESHOLD, verbose=False)
            assert not (tmp / "external_docs.txt").exists(), "external_docs.txt should NOT exist"
            assert report["internal_chunks"] > 0
            assert report["external_chunks"] == 0
            assert report["external_fraction"] == 0.0

    def test_mixed_corpus_splits_correctly(self):
        docs = [
            {"title": "External PDF paper", "source": "pdf",
             "text": _rich_text("EXTERNAL_MARKER")},
            {"title": "UltimateDocResearcher walkthrough", "source": "local",
             "text": _rich_text("INTERNAL_MARKER")},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "all_docs.txt").write_text(_make_all_docs(docs), encoding="utf-8")
            report = analyze_corpus(tmp / "all_docs.txt", output_dir=tmp,
                                    quality_threshold=self._THRESHOLD, verbose=False)
            assert report["external_chunks"] > 0
            assert report["internal_chunks"] > 0
            ext_text = (tmp / "external_docs.txt").read_text()
            assert "EXTERNAL_MARKER" in ext_text
            assert "INTERNAL_MARKER" not in ext_text

    def test_report_has_source_type_breakdown(self):
        docs = [
            {"title": "Web article", "source": "web",
             "text": _rich_text("WEB_CONTENT")},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "all_docs.txt").write_text(_make_all_docs(docs), encoding="utf-8")
            report = analyze_corpus(tmp / "all_docs.txt", output_dir=tmp,
                                    quality_threshold=self._THRESHOLD, verbose=False)
            assert "source_type_breakdown" in report
            assert "external" in report["source_type_breakdown"]
            assert "internal" in report["source_type_breakdown"]


# ── _sample_corpus_weighted ────────────────────────────────────────────────────

from autoresearch.code_suggester import _sample_corpus_weighted


class TestSampleCorpusWeighted:
    def test_uses_external_file_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            external = "EXTERNAL " * 500   # ~4000 chars
            internal = "INTERNAL " * 500
            (tmp / "all_docs_cleaned.txt").write_text(internal, encoding="utf-8")
            (tmp / "external_docs.txt").write_text(external, encoding="utf-8")

            sampled, note = _sample_corpus_weighted(
                tmp / "all_docs_cleaned.txt", max_chars=1000, external_fraction=0.70
            )
            # External should dominate
            assert sampled.count("EXTERNAL") > sampled.count("INTERNAL")
            assert "70%" in note or "external" in note.lower()

    def test_falls_back_gracefully_without_external_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus = "ALLCONTENT " * 200
            (tmp / "all_docs_cleaned.txt").write_text(corpus, encoding="utf-8")

            sampled, note = _sample_corpus_weighted(tmp / "all_docs_cleaned.txt", max_chars=500)
            assert "ALLCONTENT" in sampled
            assert "combined corpus" in note.lower() or "external" in note.lower()

    def test_respects_max_chars_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "all_docs_cleaned.txt").write_text("X" * 10_000, encoding="utf-8")
            (tmp / "external_docs.txt").write_text("E" * 10_000, encoding="utf-8")

            sampled, _ = _sample_corpus_weighted(
                tmp / "all_docs_cleaned.txt", max_chars=2000, external_fraction=0.70
            )
            assert len(sampled) <= 2500  # allow small overhead from separator strings

    def test_full_corpus_returned_when_smaller_than_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            short = "Short external content " * 5
            (tmp / "all_docs_cleaned.txt").write_text("internal context", encoding="utf-8")
            (tmp / "external_docs.txt").write_text(short, encoding="utf-8")

            sampled, _ = _sample_corpus_weighted(
                tmp / "all_docs_cleaned.txt", max_chars=50_000
            )
            assert "Short external content" in sampled


# ── eval_spec.yaml external_novelty ───────────────────────────────────────────

class TestEvalSpecExternalNovelty:
    def _load_spec(self) -> dict:
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        spec_path = ROOT / "eval" / "eval_spec.yaml"
        return yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    def test_external_novelty_criterion_present(self):
        spec = self._load_spec()
        names = [c["name"] for c in spec["criteria"]]
        assert "external_novelty" in names

    def test_external_novelty_has_required_fields(self):
        spec = self._load_spec()
        criterion = next(c for c in spec["criteria"] if c["name"] == "external_novelty")
        assert "question" in criterion
        assert "weight" in criterion
        assert "description" in criterion
        assert criterion["weight"] > 0

    def test_spec_version_updated(self):
        spec = self._load_spec()
        assert spec["version"] != "1.0", "Version should be bumped after adding external_novelty"

    def test_all_criteria_have_valid_weights(self):
        spec = self._load_spec()
        for c in spec["criteria"]:
            assert isinstance(c["weight"], (int, float))
            assert c["weight"] > 0

    def test_total_criteria_count(self):
        spec = self._load_spec()
        # Should now have 6 criteria (5 original + external_novelty)
        assert len(spec["criteria"]) == 6
