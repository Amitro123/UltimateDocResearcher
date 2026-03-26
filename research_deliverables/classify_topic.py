"""
research_deliverables/classify_topic.py
----------------------------------------
Topic classifier: maps a free-text research topic to a deliverable set
that determines which output files to generate and how to weight them.

Four deliverable sets:
  "code"    — implementation-heavy topics (APIs, SDKs, code patterns)
  "arch"    — system design topics (distributed systems, data pipelines)
  "process" — workflow / methodology topics (evals, MLOps, fine-tuning)
  "market"  — survey / landscape topics (comparisons, benchmarks, state-of-art)

Each set defines:
  - which deliverables to produce
  - which Jinja2 templates to render
  - prompt guidance passed to the generators

Keyword matching uses simple bag-of-words; extend _RULES for new categories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ResearchType = Literal["code", "arch", "process", "market",
                       "error_log", "paper", "codebase", "website"]

# Deliverable filenames (relative to the run output dir)
ALL_DELIVERABLES = [
    "SUMMARY.md",
    "ARCHITECTURE.md",
    "IMPLEMENTATION.md",
    "PLAN.md",
    "RISKS.md",
    "BENCHMARKS.md",
    "NEXT_STEPS.md",
]

# Template names (in research_deliverables/templates/)
_TEMPLATE_MAP: dict[str, str] = {
    # Existing
    "SUMMARY.md":        "summary.jinja2",
    "ARCHITECTURE.md":   "architecture.jinja2",
    "IMPLEMENTATION.md": "implementation.jinja2",
    "PLAN.md":           "plan.jinja2",
    "RISKS.md":          "risks.jinja2",
    "BENCHMARKS.md":     "benchmarks.jinja2",
    "NEXT_STEPS.md":     "next_steps.jinja2",
    # error_log input type
    "ROOT_CAUSE.md":     "root_cause.jinja2",
    "FIX_STEPS.md":      "fix_steps.jinja2",
    "PREVENTION.md":     "prevention.jinja2",
    # paper input type
    "KEY_TAKEAWAYS.md":  "key_takeaways.jinja2",
}


@dataclass
class DeliverableSet:
    research_type: ResearchType
    deliverables: list[str]          # which files to generate
    focus_hint: str                  # prompt guidance for generators
    extra_sections: dict[str, list[str]] = field(default_factory=dict)
    # extra_sections: per-deliverable list of extra section headings to include


# ── Keyword rules (ordered by priority) ──────────────────────────────────────

_RULES: list[tuple[list[str], ResearchType]] = [
    # Market/survey first — before arch keywords confuse it
    (["survey", "landscape", "compare", "comparison", "state of the art",
      "sota", "benchmark", "versus", "vs ", " vs.", "market", "overview of",
      "review of", "taxonomy"], "market"),

    # Architecture
    (["architecture", "system design", "distributed", "microservice",
      "pipeline", "data flow", "scalab", "infra", "infrastructure",
      "database", "storage", "caching", "cache layer", "stream",
      "event-driven", "event driven", "kafka", "pubsub", "pub/sub",
      "queue", "async worker", "worker pool"], "arch"),

    # Process / methodology
    (["eval", "evaluation", "fine-tun", "finetun", "rlhf", "dpo",
      "mlops", "training loop", "alignment", "red-team", "red team",
      "prompt engineering", "prompting strategy", "testing strategy",
      "deployment", "ci/cd", "observabilit", "monitor", "logging"],
     "process"),

    # Code / implementation (catch-all for code-heavy topics)
    (["sdk", "api", "code", "implement", "pattern", "snippet", "tool use",
      "tool call", "function call", "agent", "plugin", "library",
      "integration", "client", "server", "endpoint", "rest", "graphql",
      "python", "typescript", "javascript", "rust", "golang", "react",
      "langchain", "llamaindex", "rag", "retrieval"], "code"),
]


def classify_topic(topic: str) -> DeliverableSet:
    """
    Classify *topic* into one of four research types and return a
    ``DeliverableSet`` specifying which deliverables to generate.

    Falls back to "code" if no keywords match.

    Examples
    --------
    >>> classify_topic("multi-tenant RAG with Claude tool use").research_type
    'code'
    >>> classify_topic("Architecture of a streaming data pipeline").research_type
    'arch'
    >>> classify_topic("Survey of LLM evaluation frameworks 2025").research_type
    'market'
    """
    lower = topic.lower()

    matched: ResearchType = "code"  # default
    for keywords, rtype in _RULES:
        if any(kw in lower for kw in keywords):
            matched = rtype
            break

    return _build_deliverable_set(matched, topic)


def _build_deliverable_set(rtype: ResearchType, topic: str) -> DeliverableSet:
    """Build the DeliverableSet for a given research type."""

    # Phase 11: every topic type produces the full canonical package:
    #   SUMMARY + ARCHITECTURE (Mermaid) + PLAN (5-step) + RISKS (3 failure modes)
    #   + BENCHMARKS + CODE
    # The focus_hint and extra_sections tune the *content* per type.

    if rtype == "code":
        return DeliverableSet(
            research_type="code",
            deliverables=[
                "SUMMARY.md", "ARCHITECTURE.md", "IMPLEMENTATION.md",
                "RISKS.md", "BENCHMARKS.md", "NEXT_STEPS.md",
            ],
            focus_hint=(
                "Focus on concrete, copy-paste-ready code patterns, SDK usage, "
                "and implementation gotchas. Prefer working code over theory."
            ),
            extra_sections={
                "ARCHITECTURE.md": ["Component Diagram", "Data Flow", "Trade-offs"],
                "PLAN.md":         [],
                "RISKS.md":        [],
                "BENCHMARKS.md":   ["Comparison Table", "Performance Numbers",
                                    "When to Use Each Option"],
            },
        )

    if rtype == "arch":
        return DeliverableSet(
            research_type="arch",
            deliverables=[
                "SUMMARY.md", "ARCHITECTURE.md", "PLAN.md",
                "RISKS.md", "BENCHMARKS.md",
            ],
            focus_hint=(
                "Focus on component boundaries, data flows, trade-offs between "
                "design options, and scalability considerations. Use Mermaid "
                "diagrams."
            ),
            extra_sections={
                "ARCHITECTURE.md": ["Component Diagram", "Data Flow", "Trade-offs"],
                "PLAN.md":         [],
                "RISKS.md":        [],
                "BENCHMARKS.md":   ["Comparison Table", "Performance Numbers"],
            },
        )

    if rtype == "process":
        return DeliverableSet(
            research_type="process",
            deliverables=[
                "SUMMARY.md", "ARCHITECTURE.md", "IMPLEMENTATION.md",
                "RISKS.md", "BENCHMARKS.md", "NEXT_STEPS.md",
            ],
            focus_hint=(
                "Focus on step-by-step workflows, tooling choices, evaluation "
                "metrics, and operational concerns. Make each step actionable."
            ),
            extra_sections={
                "ARCHITECTURE.md": ["Component Diagram", "Data Flow", "Trade-offs"],
                "PLAN.md":         [],
                "RISKS.md":        [],
                "BENCHMARKS.md":   ["Comparison Table", "Performance Numbers",
                                    "Cost Analysis"],
            },
        )

    # "market" — survey / landscape
    return DeliverableSet(
        research_type="market",
        deliverables=[
            "SUMMARY.md", "ARCHITECTURE.md", "PLAN.md",
            "RISKS.md", "BENCHMARKS.md",
        ],
        focus_hint=(
            "Focus on comparative analysis, quantitative benchmarks, "
            "vendor/tool trade-offs, and decision criteria for choosing "
            "between options. Use tables where applicable."
        ),
        extra_sections={
            "ARCHITECTURE.md": ["Component Diagram", "Data Flow", "Trade-offs"],
            "PLAN.md":         [],
            "RISKS.md":        [],
            "BENCHMARKS.md":   ["Comparison Table", "Performance Numbers",
                                "Cost Analysis", "When to Use Each Option"],
        },
    )


def template_for(deliverable_name: str) -> str:
    """Return the Jinja2 template filename for a deliverable."""
    return _TEMPLATE_MAP.get(deliverable_name, "summary.jinja2")


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "Claude tool use patterns for agentic workflows"
    ds = classify_topic(topic)
    print(f"Topic:   {topic}")
    print(f"Type:    {ds.research_type}")
    print(f"Outputs: {', '.join(ds.deliverables)}")
    print(f"Hint:    {ds.focus_hint}")
