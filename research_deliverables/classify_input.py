"""
research_deliverables/classify_input.py
----------------------------------------
Classifies corpus INPUT TYPE based on content signals — separate from
topic classification (classify_topic.py).

Input types:
  error_log → ROOT_CAUSE.md + FIX_STEPS.md + PREVENTION.md
  codebase  → ARCHITECTURE.md + TESTS.md + CODE/
  paper     → SUMMARY.md + KEY_TAKEAWAYS.md + BENCHMARKS.md
  website   → FLOW.md + INTEGRATION.md
  text      → generic fallback (delegates to topic-based classification)

Usage:
    from research_deliverables.classify_input import classify_input, input_deliverable_set

    # Auto-detect from corpus
    itype = classify_input(corpus_text)

    # Override from CLI
    itype = classify_input(corpus_text, hint="error_log")

    # Get deliverable set
    dset = input_deliverable_set(itype, topic)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from research_deliverables.classify_topic import DeliverableSet

InputType = Literal["error_log", "codebase", "paper", "website", "text"]

_VALID_HINTS: frozenset[str] = frozenset(
    {"error_log", "codebase", "paper", "website", "text"}
)


# ── Signal patterns ────────────────────────────────────────────────────────────

_ERROR_LOG_SIGNALS: list[re.Pattern] = [
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),          # ISO timestamps
    re.compile(r"\[(ERROR|WARNING|CRITICAL|FATAL|INFO)\]"),         # log severity
    re.compile(r"httpRequest\.status=\d{3}"),                       # GCP structured log
    re.compile(r"Traceback \(most recent call last\)", re.I),       # Python traceback
    re.compile(r"at\s+\w[\w$.]+\([\w/]+\.[\w]+:\d+\)"),           # Java stack frame
    re.compile(r"QUOTA_EXCEEDED|PERMISSION_DENIED|RESOURCE_EXHAUSTED"),
    re.compile(r"(root[_ ]cause|stack[_ ]trace)", re.I),
    re.compile(r"severity\s*[=:]\s*(ERROR|WARNING|CRITICAL)", re.I),
    re.compile(r"labels\.\w+\s*=\s*\"(EMBEDDING|CHUNKING|INDEX|SCHEMA)", re.I),
    re.compile(r"pipeline_stage|error_code|exception_type", re.I),
    re.compile(r"Exception:|Error:|caused by:", re.I),
    re.compile(r"\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+(WARN|ERROR|INFO|DEBUG)", re.I),
]

_PAPER_SIGNALS: list[re.Pattern] = [
    re.compile(r"^#{1,2}\s*(abstract|introduction|conclusion|related work|references)", re.I | re.MULTILINE),
    re.compile(r"\[?\d+\]\s+[A-Z][a-z]+,\s+[A-Z]\."),             # citation style [1] Author, A.
    re.compile(r"arXiv|doi\.org|proceedings of|journal of", re.I),
    re.compile(r"\bet al\.", re.I),
    re.compile(r"(Figure|Table|Equation|Theorem)\s+\d+", re.I),
    re.compile(r"(our|we) (propose|present|evaluate|demonstrate)", re.I),
]

_CODEBASE_SIGNALS: list[re.Pattern] = [
    re.compile(r"^(def |class |import |from \w+ import)", re.MULTILINE),
    re.compile(r"^(function |const |let |var |interface |export (default )?)", re.MULTILINE),
    re.compile(r"```(python|typescript|javascript|go|rust|java|cpp|c\+\+|bash|sh)\n"),
    re.compile(r"^\s{4}(def |class )", re.MULTILINE),
    re.compile(r"(async def |await |\.then\(|\.catch\()", re.I),
    re.compile(r"(git log|git commit|git diff|git blame)", re.I),
]

_WEBSITE_SIGNALS: list[re.Pattern] = [
    re.compile(r"<(html|body|div|head|span|section|article)\b", re.I),
    re.compile(r"(GET|POST|PUT|DELETE|PATCH)\s+/[\w/]+\s+HTTP"),
    re.compile(r"<title>|<meta\s|<link rel=", re.I),
    re.compile(r"(nav|footer|header|sidebar)\s*{", re.I),   # CSS
    re.compile(r"onClick|addEventListener|querySelector", re.I),
]


def _score(text: str, patterns: list[re.Pattern]) -> int:
    """Count how many distinct patterns have at least one match in text."""
    return sum(1 for p in patterns if p.search(text))


# ── Public API ────────────────────────────────────────────────────────────────

def classify_input(corpus: str, hint: Optional[str] = None) -> InputType:
    """
    Detect the type of input document from corpus content.

    Args:
        corpus: Raw corpus text (or a representative sample of it).
        hint:   User-provided override ('error_log', 'paper', 'codebase',
                'website', 'text'). If valid, overrides auto-detection.

    Returns:
        InputType literal ('error_log' | 'codebase' | 'paper' | 'website' | 'text')
    """
    if hint is not None:
        normalized = hint.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _VALID_HINTS:
            return normalized  # type: ignore[return-value]

    sample = corpus[:10_000]  # only score first 10K chars for speed

    scores: dict[str, int] = {
        "error_log": _score(sample, _ERROR_LOG_SIGNALS),
        "paper":     _score(sample, _PAPER_SIGNALS),
        "codebase":  _score(sample, _CODEBASE_SIGNALS),
        "website":   _score(sample, _WEBSITE_SIGNALS),
    }

    best = max(scores, key=lambda k: scores[k])
    # Require at least 2 signal hits to avoid misclassification
    if scores[best] >= 2:
        return best  # type: ignore[return-value]

    return "text"


def input_deliverable_set(input_type: InputType, topic: str) -> DeliverableSet:
    """
    Build a DeliverableSet tailored to the detected input type.

    For 'text', falls back to topic-based classification.
    """
    if input_type == "error_log":
        return DeliverableSet(
            research_type="error_log",  # type: ignore[arg-type]
            deliverables=["ROOT_CAUSE.md", "FIX_STEPS.md", "PREVENTION.md"],
            focus_hint=(
                "This corpus is an error log / incident report. "
                "Focus on: what failed, why it failed, the exact commands used to fix it, "
                "and concrete monitoring / validation steps to prevent recurrence. "
                "Extract real error codes, affected counts, latency numbers, and GCP commands."
            ),
            extra_sections={
                "ROOT_CAUSE.md": ["Error Timeline", "Impact Assessment"],
                "FIX_STEPS.md":  ["Commands Reference", "Verification"],
                "PREVENTION.md": ["Monitoring & Alerting", "Validation Gates"],
            },
        )

    if input_type == "paper":
        return DeliverableSet(
            research_type="paper",  # type: ignore[arg-type]
            deliverables=["SUMMARY.md", "KEY_TAKEAWAYS.md", "BENCHMARKS.md"],
            focus_hint=(
                "This corpus is an academic paper. Extract the core contributions, "
                "experimental results, benchmark comparisons, and practical takeaways "
                "for practitioners. Quote specific numbers and model names."
            ),
            extra_sections={
                "BENCHMARKS.md": ["Comparison Table", "Performance Numbers"],
            },
        )

    if input_type == "codebase":
        return DeliverableSet(
            research_type="codebase",  # type: ignore[arg-type]
            deliverables=["ARCHITECTURE.md", "TESTS.md"],
            focus_hint=(
                "This corpus is a codebase. Focus on component boundaries, "
                "data flow through the system, test coverage gaps, and refactoring "
                "opportunities. Use Mermaid diagrams for the architecture."
            ),
            extra_sections={
                "ARCHITECTURE.md": ["Component Diagram", "Data Flow", "Trade-offs"],
            },
        )

    if input_type == "website":
        return DeliverableSet(
            research_type="website",  # type: ignore[arg-type]
            deliverables=["FLOW.md", "INTEGRATION.md"],
            focus_hint=(
                "This corpus is website content. Focus on user flows, "
                "integration points with external services, and API surfaces."
            ),
            extra_sections={},
        )

    # "text" — fall back to topic-based classification
    from research_deliverables.classify_topic import classify_topic
    return classify_topic(topic)
