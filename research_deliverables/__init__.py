"""
research_deliverables/
----------------------
Multi-format research output system for UltimateDocResearcher.

Given a research corpus and topic, generates a structured package of
deliverables tailored to the research type (code, architecture, process, market).

Output per run: results/<slug>-<run_id>/
  SUMMARY.md          — executive overview
  ARCHITECTURE.md     — system design patterns & diagrams
  CODE/               — code_suggestions.md (from code_suggester.py)
  IMPLEMENTATION.md   — step-by-step implementation plan
  RISKS.md            — risk register with mitigations
  BENCHMARKS.md       — performance data and comparisons
  NEXT_STEPS.md       — prioritised action items

Usage:
    python -m autoresearch.research --topic "multi-tenant RAG" --iterations 1
"""
from research_deliverables.classify_topic import classify_topic, DeliverableSet
from research_deliverables.generators import generate_deliverables

__all__ = ["classify_topic", "DeliverableSet", "generate_deliverables"]
