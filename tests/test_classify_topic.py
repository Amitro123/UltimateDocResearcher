"""
tests/test_classify_topic.py
-----------------------------
Unit tests for research_deliverables/classify_topic.py.

Covers:
  - classify_topic: all four research types (market, arch, process, code)
  - Fallback to "code" when no keywords match
  - Case-insensitive keyword matching
  - Priority order: market before arch when both keywords present
  - DeliverableSet contents: non-empty deliverables, focus_hint
  - All returned deliverables exist in ALL_DELIVERABLES
  - template_for: known and unknown deliverable names
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_deliverables.classify_topic import (
    classify_topic,
    template_for,
    ALL_DELIVERABLES,
    DeliverableSet,
)


# ── classify_topic ────────────────────────────────────────────────────────────

class TestClassifyTopicMarket(unittest.TestCase):

    def test_survey_keyword(self):
        ds = classify_topic("Survey of LLM evaluation frameworks 2025")
        self.assertEqual(ds.research_type, "market")

    def test_compare_keyword(self):
        ds = classify_topic("Compare GPT-4 vs Claude 3.5 for code generation")
        self.assertEqual(ds.research_type, "market")

    def test_benchmark_keyword(self):
        ds = classify_topic("Benchmark results for open-source embedding models")
        self.assertEqual(ds.research_type, "market")

    def test_landscape_keyword(self):
        ds = classify_topic("AI agent framework landscape overview")
        self.assertEqual(ds.research_type, "market")

    def test_market_type_deliverables(self):
        ds = classify_topic("SOTA vector databases comparison")
        self.assertIn("SUMMARY.md", ds.deliverables)
        self.assertIn("BENCHMARKS.md", ds.deliverables)


class TestClassifyTopicArch(unittest.TestCase):

    def test_architecture_keyword(self):
        ds = classify_topic("Architecture of a streaming data pipeline")
        self.assertEqual(ds.research_type, "arch")

    def test_distributed_keyword(self):
        ds = classify_topic("Distributed inference for large language models")
        self.assertEqual(ds.research_type, "arch")

    def test_microservice_keyword(self):
        ds = classify_topic("Microservice design for a multi-tenant SaaS platform")
        self.assertEqual(ds.research_type, "arch")

    def test_pipeline_keyword(self):
        ds = classify_topic("Data pipeline for real-time ML feature engineering")
        self.assertEqual(ds.research_type, "arch")

    def test_arch_includes_risks(self):
        ds = classify_topic("Cache layer design patterns")
        self.assertIn("RISKS.md", ds.deliverables)
        self.assertIn("ARCHITECTURE.md", ds.deliverables)


class TestClassifyTopicProcess(unittest.TestCase):

    def test_eval_keyword(self):
        ds = classify_topic("LLM evaluation best practices")
        self.assertEqual(ds.research_type, "process")

    def test_finetune_keyword(self):
        ds = classify_topic("Fine-tuning strategies for domain adaptation")
        self.assertEqual(ds.research_type, "process")

    def test_mlops_keyword(self):
        # "pipeline" is an arch keyword and arch rules are checked before process,
        # so "MLOps pipelines" → arch. Use a topic with only process keywords.
        ds = classify_topic("MLOps evaluation and monitoring strategy")
        self.assertEqual(ds.research_type, "process")

    def test_deployment_keyword(self):
        ds = classify_topic("Deployment strategies for multi-region LLM APIs")
        self.assertEqual(ds.research_type, "process")

    def test_process_includes_implementation(self):
        ds = classify_topic("Prompt engineering best practices")
        self.assertIn("IMPLEMENTATION.md", ds.deliverables)


class TestClassifyTopicCode(unittest.TestCase):

    def test_rag_keyword(self):
        ds = classify_topic("RAG patterns with LangChain")
        self.assertEqual(ds.research_type, "code")

    def test_sdk_keyword(self):
        ds = classify_topic("Anthropic Python SDK tool use patterns")
        self.assertEqual(ds.research_type, "code")

    def test_python_keyword(self):
        ds = classify_topic("Python async patterns for LLM agents")
        self.assertEqual(ds.research_type, "code")

    def test_agent_keyword(self):
        ds = classify_topic("Building a multi-step agent with Claude")
        self.assertEqual(ds.research_type, "code")

    def test_code_includes_next_steps(self):
        ds = classify_topic("REST API patterns for AI endpoints")
        self.assertIn("NEXT_STEPS.md", ds.deliverables)
        self.assertIn("IMPLEMENTATION.md", ds.deliverables)


class TestClassifyTopicFallback(unittest.TestCase):

    def test_no_keywords_returns_code(self):
        ds = classify_topic("random unrelated words here")
        self.assertEqual(ds.research_type, "code")

    def test_empty_string_returns_code(self):
        ds = classify_topic("")
        self.assertEqual(ds.research_type, "code")

    def test_single_word_no_match(self):
        ds = classify_topic("cooking")
        self.assertEqual(ds.research_type, "code")


class TestClassifyTopicCaseInsensitive(unittest.TestCase):

    def test_uppercase_keyword(self):
        ds = classify_topic("SURVEY OF RAG SYSTEMS")
        self.assertEqual(ds.research_type, "market")

    def test_mixed_case_architecture(self):
        ds = classify_topic("ARCHITECTURE of a Distributed System")
        self.assertEqual(ds.research_type, "arch")

    def test_mixed_case_code(self):
        ds = classify_topic("Python SDK Integration")
        self.assertEqual(ds.research_type, "code")


class TestClassifyTopicPriority(unittest.TestCase):

    def test_market_before_arch(self):
        # "pipeline" is an arch keyword, but "survey" is market → market wins
        ds = classify_topic("Survey of data pipeline architectures")
        self.assertEqual(ds.research_type, "market")

    def test_market_before_code(self):
        # "sdk" is code keyword, but "comparison" is market → market wins
        ds = classify_topic("SDK comparison and benchmark")
        self.assertEqual(ds.research_type, "market")

    def test_arch_before_process(self):
        # "deployment" is process, "architecture" is arch → arch wins
        ds = classify_topic("Architecture for zero-downtime deployment")
        self.assertEqual(ds.research_type, "arch")


class TestDeliverableSetContents(unittest.TestCase):

    def test_deliverables_are_nonempty(self):
        for topic in [
            "code patterns", "architecture design",
            "evaluation workflow", "survey of models"
        ]:
            ds = classify_topic(topic)
            self.assertGreater(len(ds.deliverables), 0, f"Empty deliverables for: {topic}")

    def test_all_deliverables_in_allowed_set(self):
        for topic in [
            "SDK tool use", "distributed architecture",
            "fine-tuning process", "benchmark survey"
        ]:
            ds = classify_topic(topic)
            for d in ds.deliverables:
                self.assertIn(d, ALL_DELIVERABLES, f"{d} not in ALL_DELIVERABLES")

    def test_summary_always_included(self):
        for topic in [
            "code patterns", "architecture design",
            "evaluation workflow", "survey of models"
        ]:
            ds = classify_topic(topic)
            self.assertIn("SUMMARY.md", ds.deliverables, f"SUMMARY.md missing for: {topic}")

    def test_focus_hint_is_nonempty_string(self):
        for rtype_topic in [
            "RAG code patterns",
            "distributed architecture design",
            "MLOps evaluation workflow",
            "benchmark survey comparison",
        ]:
            ds = classify_topic(rtype_topic)
            self.assertIsInstance(ds.focus_hint, str)
            self.assertGreater(len(ds.focus_hint.strip()), 20)

    def test_returns_deliverable_set_instance(self):
        ds = classify_topic("some topic")
        self.assertIsInstance(ds, DeliverableSet)

    def test_research_type_is_valid_literal(self):
        valid = {"code", "arch", "process", "market"}
        for topic in ["code", "architecture", "evaluation", "survey"]:
            ds = classify_topic(topic)
            self.assertIn(ds.research_type, valid)


# ── template_for ──────────────────────────────────────────────────────────────

class TestTemplateFor(unittest.TestCase):

    def test_summary_template(self):
        self.assertEqual(template_for("SUMMARY.md"), "summary.jinja2")

    def test_architecture_template(self):
        self.assertEqual(template_for("ARCHITECTURE.md"), "architecture.jinja2")

    def test_implementation_template(self):
        self.assertEqual(template_for("IMPLEMENTATION.md"), "implementation.jinja2")

    def test_risks_template(self):
        self.assertEqual(template_for("RISKS.md"), "risks.jinja2")

    def test_benchmarks_template(self):
        self.assertEqual(template_for("BENCHMARKS.md"), "benchmarks.jinja2")

    def test_next_steps_template(self):
        self.assertEqual(template_for("NEXT_STEPS.md"), "next_steps.jinja2")

    def test_unknown_defaults_to_summary(self):
        self.assertEqual(template_for("UNKNOWN.md"), "summary.jinja2")

    def test_all_deliverables_have_templates(self):
        for d in ALL_DELIVERABLES:
            t = template_for(d)
            self.assertTrue(t.endswith(".jinja2"), f"{d} → {t} should be a .jinja2 file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
