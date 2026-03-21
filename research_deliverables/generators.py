"""
research_deliverables/generators.py
-------------------------------------
LLM-powered generators for each deliverable type.

Each generator:
  1. Builds a focused prompt from the corpus extract + topic + focus_hint
  2. Calls the LLM (same llm_client used by code_suggester)
  3. Renders the Jinja2 template with the structured response
  4. Returns the final Markdown string

All generators share the same _generate() helper that handles LLM calls
and graceful fallback (if the LLM is unavailable, returns a stub with
the raw LLM response or an explanatory placeholder).

The entry point is generate_deliverables(), which:
  - Classifies the topic
  - Runs code_suggester for the CODE/ deliverable
  - Calls the appropriate generators for each deliverable in the set
  - Writes all files to the output directory
  - Returns a DeliverablePackage with metadata
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── DeliverablePackage ────────────────────────────────────────────────────────

@dataclass
class DeliverablePackage:
    run_id: str
    topic: str
    research_type: str
    output_dir: Path
    files: dict[str, Path]     # deliverable name → path
    metadata: dict             # corpus stats, model, timestamps, etc.
    errors: dict[str, str] = field(default_factory=dict)  # name → error msg


# ── Jinja2 rendering ──────────────────────────────────────────────────────────

def _render_template(template_name: str, context: dict) -> str:
    """Render a Jinja2 template from research_deliverables/templates/."""
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
        templates_dir = Path(__file__).resolve().parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        tpl = env.get_template(template_name)
        return tpl.render(**context)
    except Exception as exc:
        # Fall back to a simple key=value dump
        lines = [f"# {context.get('topic', 'Research Output')}\n"]
        for k, v in context.items():
            if isinstance(v, str) and v.strip():
                lines.append(f"## {k.replace('_', ' ').title()}\n\n{v}\n")
        return "\n".join(lines)


# ── LLM call helper ───────────────────────────────────────────────────────────

def _llm_chat(system: str, user: str, model: Optional[str] = None,
              max_tokens: int = 2048, use_cache: bool = True) -> str:
    """Call the LLM. Returns plain string response."""
    try:
        from autoresearch.llm_client import chat, best_available_model
        effective_model = model or best_available_model()
        return chat(
            messages=[{"role": "user", "content": user}],
            model=effective_model,
            system=system,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )
    except Exception as exc:
        return f"[LLM unavailable: {exc}]\n\n_Run with a configured LLM to generate this section._"


# ── Structured extraction ─────────────────────────────────────────────────────

_SECTION_RE = re.compile(
    r"##\s+(?P<heading>[^\n]+)\n(?P<body>.*?)(?=\n##\s|\Z)",
    re.DOTALL,
)

def _extract_sections(text: str) -> dict[str, str]:
    """Parse ## heading / body pairs from LLM output into a dict.

    Also stores positional keys '_section_0', '_section_1', … so callers
    can fall back to positional extraction when heading names don't match.
    """
    sections: dict[str, str] = {}
    idx = 0
    for m in _SECTION_RE.finditer(text):
        key = m.group("heading").strip().lower().replace(" ", "_")
        body = m.group("body").strip()
        sections[key] = body
        sections[f"_section_{idx}"] = body  # positional alias
        idx += 1
    return sections


def _extract_or_default(sections: dict[str, str], *keys: str,
                        default: str = "_No content generated._") -> str:
    """Return the first matching section key, or default."""
    for k in keys:
        norm = k.lower().replace(" ", "_")
        if norm in sections:
            return sections[norm]
    return default


# ── Per-deliverable system prompts ────────────────────────────────────────────

_SUMMARY_SYSTEM = """\
You are a technical research analyst. Given a research corpus extract, produce a
concise executive summary for a developer audience. Target ~300 words total.

Structure your response with these exact ## headings:
## Overview
(2-3 sentences: what the topic is, why it matters right now)

## Key Findings
(5-7 bullet points, each a single crisp sentence — the most actionable insights)

## Recommended Next Action
(1 sentence: the single highest-leverage thing to do with these findings)

Be specific. Reference actual techniques, libraries, or metric names from the corpus.
No filler phrases. Aim for density: every sentence must carry new information.
"""

_ARCHITECTURE_SYSTEM = """\
You are a senior software architect. Given a research corpus extract, produce a
concise architecture document with a Mermaid diagram.

Structure your response with these exact ## headings:
## System Overview
(2-3 sentences describing the overall system architecture)

## Component Diagram
(A valid Mermaid flowchart — wrap it in ```mermaid ... ``` fences.
Use flowchart LR or TD. Label edges. Show all major components and their
relationships. Example skeleton:
```mermaid
flowchart LR
    A[Collector] -->|raw docs| B[Analyzer]
    B -->|cleaned corpus| C[LLM]
    C -->|suggestions| D[Results]
```
)

## Components
(Numbered list: component name — one-line responsibility — key interface/API)

## Data Flow
(Step-by-step: how data enters, transforms, and exits the system)

## Trade-offs
(3-5 bullet points: what this architecture optimises for vs. what it sacrifices)

## Design Decisions
(2-3 key decisions with brief rationale — what alternatives were rejected)

Use concrete component names from the corpus. The Mermaid diagram must be valid syntax.
"""

_IMPLEMENTATION_SYSTEM = """\
You are a senior engineer writing an implementation guide. Given a research corpus
extract, produce a step-by-step implementation plan.

Structure your response with these exact ## headings:
## Prerequisites
(Bulleted list: tools, libraries, knowledge required before starting)

## Step-by-Step Plan
(Numbered list of concrete implementation steps with brief description of each)

## Key APIs
(Short descriptions of the most important APIs/SDKs to use, with example calls)

## Code Patterns
(2-3 patterns from the research that should be applied, with mini code examples)

## Common Pitfalls
(3-5 gotchas or mistakes to avoid, drawn from the research)

## Validation Checklist
(Bullet list of things to verify after each major step)

## Dependencies & Tooling
(pip/npm packages needed, with version pins where important)

Be specific. Use real API names and function signatures from the corpus.
"""

_RISKS_SYSTEM = """\
You are a technical risk analyst. Given a research corpus, identify the 3 most
critical failure modes and produce a focused risk register.

Structure your response with these exact ## headings:
## Risk Summary
(1 sentence: the dominant risk theme across all three failure modes)

## Risk Register
(Markdown table with exactly 3 rows — the 3 most critical failure modes.
Columns: Failure Mode | Likelihood | Impact | Mitigation
Use High/Medium/Low for Likelihood and Impact.
Each "Failure Mode" cell must name a concrete, specific failure — not a generic label.)

## Mitigation Priorities
(Ordered list 1-3: for each failure mode, one concrete mitigation step with
the team/role responsible and a measurable success criterion)

Be concrete. Name specific failure modes tied to the research topic.
"""

_BENCHMARKS_SYSTEM = """\
You are a technical analyst producing a benchmark and comparison report. Given a
research corpus extract, produce a data-driven comparison.

Structure your response with these exact ## headings:
## Summary
(2-3 sentences: what was compared and the headline finding)

## Comparison Table
(Markdown table comparing options on: Performance, Cost, Ease of Use, Maturity, Use Case Fit)

## Performance Numbers
(Concrete metrics from the research: latency, throughput, accuracy scores, etc.)

## Cost Analysis
(Relative cost comparison: compute, API costs, engineering effort)

## When to Use Each Option
(Bullet list: option name → best use case)

## Recommendations
(Clear recommendation for the most common use case, with reasoning)

Use numbers from the corpus where available. Label estimates clearly as "~approx".
"""

_NEXT_STEPS_SYSTEM = """\
You are a senior engineer writing a prioritised action plan. Given a research
corpus extract and topic, produce concrete next steps.

Structure your response with these exact ## headings:
## Immediate Actions (This Week)
(2-3 actions that can be done right now, each with a clear deliverable)

## Short-Term (1–4 Weeks)
(3-5 actions for the near term, ordered by priority)

## Medium-Term (1–3 Months)
(2-3 larger initiatives, with rough effort estimates)

## Quick Wins
(2-3 easy improvements with high payoff)

## Medium-term Improvements
(2-3 more substantial changes worth the investment)

## Success Metrics
(Bulleted list: how to know if the implementation is successful — use measurable metrics)

Be specific. Each action should have a clear owner, deliverable, and success criterion.
"""

_PLAN_SYSTEM = """\
You are a senior engineering lead writing a 5-step rollout plan. Given a research
corpus extract, produce a concrete, time-boxed implementation roadmap.

Structure your response with these exact ## headings:
## 5-Step Rollout
(Exactly 5 numbered steps. Each step must have:
  - A bold title (e.g. **Step 1: Baseline instrumentation**)
  - 1-2 sentences describing what is built/done
  - A concrete deliverable (e.g. "merged PR", "green CI badge", "deployed endpoint")
Steps should build on each other and go from least to most risky.)

## Timeline Estimate
(Table: Step | Effort | Owner Role
Use T-shirt sizes: XS=<1 day, S=1-3 days, M=1 week, L=2-3 weeks)

## Dependencies
(Bullet list: external services, libraries, or team approvals needed before starting)

## Success Criteria
(Bullet list of measurable exit criteria for the full rollout —
quantified where possible: latency target, accuracy threshold, error rate, etc.)

## Rollback Strategy
(2-3 sentences: how to revert if step 3 or 4 fails — be specific about what breaks)

No vague advice. Every step must have a clear start/end state.
"""

_ROOT_CAUSE_SYSTEM = """\
You are a senior SRE performing a post-incident root cause analysis.
Given an error log or incident report, produce a complete Markdown document
with EXACTLY these four sections (use these exact ## headings):

## Incident Overview
2-3 sentences: what service failed, when, and the headline impact.

## Root Causes Identified
Numbered list. Each item: error code/message → trigger → why it was not caught.

## Error Timeline
Bullet list of timestamped events in chronological order.
Format each bullet: `HH:MM UTC — [SEVERITY] description`

## Impact Assessment
Markdown table with columns: Metric | Baseline | During Incident | After Fix
Use real numbers from the log.

Be concrete. Quote exact error codes and numbers. Do not add extra sections.
"""

_FIX_STEPS_SYSTEM = """\
You are a senior engineer writing a remediation runbook.
Given an error log or incident report, produce a complete Markdown document
with EXACTLY these four sections (use these exact ## headings):

## Immediate Actions
Bullet list of actions to stop the bleeding right now.
Include who should do each action and the expected time-to-impact.

## Step-by-Step Remediation
Numbered steps in the order they were executed.
Each step: **bold title**, 1-sentence rationale, exact command or config change.

## Commands Reference
Fenced code block(s) with all shell/gcloud/API commands.
Add a one-line comment above each command explaining what it does.

## Verification
Bullet list: how to confirm each root cause is resolved.
Include the exact metric or log query and its expected value after the fix.

Use real commands from the log. Prefer copy-paste code blocks. Do not add extra sections.
"""

_PREVENTION_SYSTEM = """\
You are a senior SRE writing a post-incident prevention document.
Given an incident report, produce a complete Markdown document
with EXACTLY these five sections (use these exact ## headings):

## Prevention Summary
1-2 sentences: what class of problem this was and the dominant prevention theme.

## Monitoring & Alerting
Numbered list of alerts to create.
Each entry: metric name/query → threshold → who gets paged → severity.

## Validation Gates
Numbered list of automated checks to add to the pipeline/CI/import job.
Each gate: what assertion to add and what it catches.

## Policy Changes
Bullet list: what to change → new value or policy.
Cover config, IAM, schema, and process changes.

## Runbook
Numbered steps for on-call: what to check first, diagnostic commands, escalation path.

Use specific metric names, IAM roles, and field names from the corpus.
Do not add extra sections.
"""

_KEY_TAKEAWAYS_SYSTEM = """\
You are a technical analyst distilling an academic paper for practitioners.

Structure your response with these exact ## headings:
## Core Contributions
(Numbered list: the 3-5 main claims or techniques introduced in the paper.
Each entry: one sentence stating the contribution and its significance.)

## Practical Takeaways
(Bullet list: what a practitioner should actually *do* differently based on this paper.
Focus on techniques, thresholds, configs, or design choices they can adopt today.)

## Limitations & Future Work
(Bullet list: what the authors acknowledge as limitations, plus open questions
that the paper does not answer.)

Be specific. Reference model names, dataset names, and numbers from the paper.
"""

_SYSTEM_PROMPTS = {
    "SUMMARY.md":        _SUMMARY_SYSTEM,
    "ARCHITECTURE.md":   _ARCHITECTURE_SYSTEM,
    "IMPLEMENTATION.md": _IMPLEMENTATION_SYSTEM,
    "PLAN.md":           _PLAN_SYSTEM,
    "RISKS.md":          _RISKS_SYSTEM,
    "BENCHMARKS.md":     _BENCHMARKS_SYSTEM,
    "NEXT_STEPS.md":     _NEXT_STEPS_SYSTEM,
    # Input-type specific
    "ROOT_CAUSE.md":     _ROOT_CAUSE_SYSTEM,
    "FIX_STEPS.md":      _FIX_STEPS_SYSTEM,
    "PREVENTION.md":     _PREVENTION_SYSTEM,
    "KEY_TAKEAWAYS.md":  _KEY_TAKEAWAYS_SYSTEM,
}


# ── Individual generators ─────────────────────────────────────────────────────

def _user_prompt(topic: str, corpus_extract: str, source_note: str,
                 focus_hint: str, extra_guidance: str = "") -> str:
    return textwrap.dedent(f"""\
        Research topic: {topic}

        {source_note}
        {focus_hint}
        {"" if not extra_guidance else extra_guidance + chr(10)}
        --- Corpus extract ({len(corpus_extract):,} chars) ---
        {corpus_extract}
        --- End of extract ---
    """).strip()


def generate_summary(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating SUMMARY.md …")
    user = _user_prompt(topic, corpus_extract, source_note,
                        deliverable_set.focus_hint)
    raw = _llm_chat(_SUMMARY_SYSTEM, user, model=model, max_tokens=1024)
    sections = _extract_sections(raw)

    from research_deliverables.classify_topic import template_for
    return _render_template(template_for("SUMMARY.md"), {
        "topic":              topic,
        "timestamp":          corpus_stats.get("timestamp", ""),
        "research_type":      deliverable_set.research_type,
        "corpus_chunks":      corpus_stats.get("chunks", "?"),
        "corpus_chars":       corpus_stats.get("chars", "?"),
        "run_id":             run_id,
        "overview":           _extract_or_default(sections, "overview"),
        "key_findings":       _extract_or_default(sections, "key_findings"),
        "recommended_action": _extract_or_default(sections, "recommended_next_action",
                                                   "recommended_action"),
        "deliverables":       deliverable_set.deliverables,
    })


def generate_architecture(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating ARCHITECTURE.md …")
    user = _user_prompt(topic, corpus_extract, source_note,
                        deliverable_set.focus_hint)
    raw = _llm_chat(_ARCHITECTURE_SYSTEM, user, model=model, max_tokens=2048)
    sections = _extract_sections(raw)
    extra = deliverable_set.extra_sections.get("ARCHITECTURE.md", [])

    from research_deliverables.classify_topic import template_for
    return _render_template(template_for("ARCHITECTURE.md"), {
        "topic":             topic,
        "timestamp":         corpus_stats.get("timestamp", ""),
        "run_id":            run_id,
        "system_overview":   _extract_or_default(sections, "system_overview"),
        "component_diagram": _extract_or_default(sections, "component_diagram", default=""),
        "components":        _extract_or_default(sections, "components"),
        "data_flow":         _extract_or_default(sections, "data_flow"),
        "tradeoffs":         _extract_or_default(sections, "trade-offs", "tradeoffs",
                                                  "trade_offs"),
        "design_decisions":  _extract_or_default(sections, "design_decisions"),
        "extra_sections":    {s: _extract_or_default(sections, s.lower().replace(" ", "_"))
                              for s in extra},
    })


def generate_implementation(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating IMPLEMENTATION.md …")
    extra_hint = "\n".join(
        f"Include a section titled '{s}'."
        for s in deliverable_set.extra_sections.get("IMPLEMENTATION.md", [])
    )
    user = _user_prompt(topic, corpus_extract, source_note,
                        deliverable_set.focus_hint, extra_hint)
    raw = _llm_chat(_IMPLEMENTATION_SYSTEM, user, model=model, max_tokens=3000)
    sections = _extract_sections(raw)
    extra = deliverable_set.extra_sections.get("IMPLEMENTATION.md", [])

    from research_deliverables.classify_topic import template_for
    return _render_template(template_for("IMPLEMENTATION.md"), {
        "topic":                topic,
        "timestamp":            corpus_stats.get("timestamp", ""),
        "run_id":               run_id,
        "estimated_effort":     _extract_or_default(sections, "estimated_effort",
                                                     default="See steps below"),
        "prerequisites":        _extract_or_default(sections, "prerequisites"),
        "steps":                _extract_or_default(sections, "step-by-step_plan",
                                                     "step_by_step_plan", "steps"),
        "validation_checklist": _extract_or_default(sections, "validation_checklist"),
        "dependencies":         _extract_or_default(sections, "dependencies_&_tooling",
                                                     "dependencies_and_tooling",
                                                     "dependencies"),
        "extra_sections":       {s: _extract_or_default(sections, s.lower().replace(" ", "_"))
                                 for s in extra},
    })


def generate_risks(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating RISKS.md …")
    user = _user_prompt(topic, corpus_extract, source_note,
                        deliverable_set.focus_hint)
    raw = _llm_chat(_RISKS_SYSTEM, user, model=model, max_tokens=2048)
    sections = _extract_sections(raw)
    extra = deliverable_set.extra_sections.get("RISKS.md", [])

    from research_deliverables.classify_topic import template_for
    return _render_template(template_for("RISKS.md"), {
        "topic":                 topic,
        "timestamp":             corpus_stats.get("timestamp", ""),
        "run_id":                run_id,
        "risk_summary":          _extract_or_default(sections, "risk_summary"),
        "risk_register":         _extract_or_default(sections, "risk_register"),
        "mitigation_priorities": _extract_or_default(sections, "mitigation_priorities"),
        "extra_sections":        {s: _extract_or_default(sections, s.lower().replace(" ", "_"))
                                  for s in extra},
    })


def generate_benchmarks(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating BENCHMARKS.md …")
    user = _user_prompt(topic, corpus_extract, source_note,
                        deliverable_set.focus_hint)
    raw = _llm_chat(_BENCHMARKS_SYSTEM, user, model=model, max_tokens=2048)
    sections = _extract_sections(raw)
    extra = deliverable_set.extra_sections.get("BENCHMARKS.md", [])

    from research_deliverables.classify_topic import template_for
    return _render_template(template_for("BENCHMARKS.md"), {
        "topic":               topic,
        "timestamp":           corpus_stats.get("timestamp", ""),
        "run_id":              run_id,
        "summary":             _extract_or_default(sections, "summary"),
        "comparison_table":    _extract_or_default(sections, "comparison_table"),
        "performance_numbers": _extract_or_default(sections, "performance_numbers"),
        "recommendations":     _extract_or_default(sections, "recommendations"),
        "extra_sections":      {s: _extract_or_default(sections, s.lower().replace(" ", "_"))
                                for s in extra},
    })


def generate_next_steps(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating NEXT_STEPS.md …")
    extra_hint = "\n".join(
        f"Include a section titled '{s}'."
        for s in deliverable_set.extra_sections.get("NEXT_STEPS.md", [])
    )
    user = _user_prompt(topic, corpus_extract, source_note,
                        deliverable_set.focus_hint, extra_hint)
    raw = _llm_chat(_NEXT_STEPS_SYSTEM, user, model=model, max_tokens=2048)
    sections = _extract_sections(raw)
    extra = deliverable_set.extra_sections.get("NEXT_STEPS.md", [])

    from research_deliverables.classify_topic import template_for
    return _render_template(template_for("NEXT_STEPS.md"), {
        "topic":            topic,
        "timestamp":        corpus_stats.get("timestamp", ""),
        "run_id":           run_id,
        "immediate_actions":_extract_or_default(sections, "immediate_actions_(this_week)",
                                                 "immediate_actions"),
        "short_term":       _extract_or_default(sections, "short-term_(1–4_weeks)",
                                                 "short_term"),
        "medium_term":      _extract_or_default(sections, "medium-term_(1–3_months)",
                                                 "medium_term"),
        "success_metrics":  _extract_or_default(sections, "success_metrics"),
        "extra_sections":   {s: _extract_or_default(sections, s.lower().replace(" ", "_"))
                             for s in extra},
    })


def generate_plan(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating PLAN.md …")
    user = _user_prompt(topic, corpus_extract, source_note,
                        deliverable_set.focus_hint)
    raw = _llm_chat(_PLAN_SYSTEM, user, model=model, max_tokens=2048)
    sections = _extract_sections(raw)

    from research_deliverables.classify_topic import template_for
    return _render_template(template_for("PLAN.md"), {
        "topic":            topic,
        "timestamp":        corpus_stats.get("timestamp", ""),
        "run_id":           run_id,
        "steps":            _extract_or_default(sections, "5-step_rollout",
                                                 "5_step_rollout", "steps"),
        "timeline":         _extract_or_default(sections, "timeline_estimate",
                                                 "timeline"),
        "dependencies":     _extract_or_default(sections, "dependencies"),
        "success_criteria": _extract_or_default(sections, "success_criteria"),
        "rollback":         _extract_or_default(sections, "rollback_strategy",
                                                 "rollback"),
    })


def _render_raw_doc(
    deliverable_name: str,
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    system_prompt: str,
    model: Optional[str],
    template_slots: dict[str, str],
    expected_sections: list[str],
) -> str:
    """
    Generic full-document generator for input-type deliverables.

    Asks the LLM to produce a complete Markdown document with specific ##
    headings, then extracts sections by heading name (with positional fallback).
    The extracted sections are rendered into a Jinja2 template.

    Args:
        deliverable_name:  e.g. 'ROOT_CAUSE.md'
        template_slots:    dict mapping template variable → list of heading
                           key aliases to try in order.
        expected_sections: list of the ## headings the system prompt asks for,
                           in order (used to build the reminder in the user msg).
    """
    heading_list = "\n".join(f"## {h}" for h in expected_sections)
    reminder = (
        f"Produce a complete Markdown document using EXACTLY these "
        f"{len(expected_sections)} ## headings in this order:\n{heading_list}\n"
        "Do NOT add extra headings or change the heading text."
    )
    user = _user_prompt(topic, corpus_extract, source_note,
                        deliverable_set.focus_hint, reminder)
    # Bypass prompt cache: different system prompts on the same corpus must not
    # collide with cached SUMMARY responses.
    raw = _llm_chat(system_prompt, user, model=model, max_tokens=2048,
                    use_cache=False)
    sections = _extract_sections(raw)

    from research_deliverables.classify_topic import template_for
    context: dict = {
        "topic":     topic,
        "timestamp": corpus_stats.get("timestamp", ""),
        "run_id":    run_id,
    }
    for slot_name, aliases in template_slots.items():
        context[slot_name] = _extract_or_default(sections, *aliases)

    return _render_template(template_for(deliverable_name), context)


def generate_root_cause(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating ROOT_CAUSE.md ...")
    return _render_raw_doc(
        deliverable_name="ROOT_CAUSE.md",
        topic=topic,
        corpus_extract=corpus_extract,
        source_note=source_note,
        deliverable_set=deliverable_set,
        run_id=run_id,
        corpus_stats=corpus_stats,
        system_prompt=_ROOT_CAUSE_SYSTEM,
        model=model,
        expected_sections=[
            "Incident Overview",
            "Root Causes Identified",
            "Error Timeline",
            "Impact Assessment",
        ],
        template_slots={
            "incident_overview": ["incident_overview", "overview", "summary",
                                  "_section_0"],
            "root_causes":       ["root_causes_identified", "root_causes",
                                  "key_findings", "_section_1"],
            "error_timeline":    ["error_timeline", "timeline", "_section_2"],
            "impact":            ["impact_assessment", "impact",
                                  "recommended_next_action", "_section_3"],
        },
    )


def generate_fix_steps(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating FIX_STEPS.md ...")
    return _render_raw_doc(
        deliverable_name="FIX_STEPS.md",
        topic=topic,
        corpus_extract=corpus_extract,
        source_note=source_note,
        deliverable_set=deliverable_set,
        run_id=run_id,
        corpus_stats=corpus_stats,
        system_prompt=_FIX_STEPS_SYSTEM,
        model=model,
        expected_sections=[
            "Immediate Actions",
            "Step-by-Step Remediation",
            "Commands Reference",
            "Verification",
        ],
        template_slots={
            "immediate_actions": ["immediate_actions", "overview", "_section_0"],
            "remediation_steps": ["step-by-step_remediation",
                                   "step_by_step_remediation",
                                   "key_findings", "_section_1"],
            "commands":          ["commands_reference", "commands", "_section_2"],
            "verification":      ["verification", "recommended_next_action",
                                   "_section_3"],
        },
    )


def generate_prevention(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating PREVENTION.md ...")
    return _render_raw_doc(
        deliverable_name="PREVENTION.md",
        topic=topic,
        corpus_extract=corpus_extract,
        source_note=source_note,
        deliverable_set=deliverable_set,
        run_id=run_id,
        corpus_stats=corpus_stats,
        system_prompt=_PREVENTION_SYSTEM,
        model=model,
        expected_sections=[
            "Prevention Summary",
            "Monitoring & Alerting",
            "Validation Gates",
            "Policy Changes",
            "Runbook",
        ],
        template_slots={
            "prevention_summary": ["prevention_summary", "overview", "_section_0"],
            "monitoring":         ["monitoring_&_alerting", "monitoring_and_alerting",
                                    "monitoring", "key_findings", "_section_1"],
            "validation":         ["validation_gates", "validation", "_section_2"],
            "policy_changes":     ["policy_changes", "policies", "_section_3"],
            "runbook":            ["runbook", "recommended_next_action", "_section_4"],
        },
    )


def generate_key_takeaways(
    topic: str,
    corpus_extract: str,
    source_note: str,
    deliverable_set,
    run_id: str,
    corpus_stats: dict,
    model: Optional[str] = None,
) -> str:
    print("[deliverables] Generating KEY_TAKEAWAYS.md ...")
    user = _user_prompt(topic, corpus_extract, source_note, deliverable_set.focus_hint)
    raw = _llm_chat(_KEY_TAKEAWAYS_SYSTEM, user, model=model, max_tokens=2048)
    sections = _extract_sections(raw)

    from research_deliverables.classify_topic import template_for
    return _render_template(template_for("KEY_TAKEAWAYS.md"), {
        "topic":               topic,
        "timestamp":           corpus_stats.get("timestamp", ""),
        "run_id":              run_id,
        "core_contributions":  _extract_or_default(sections, "core_contributions"),
        "practical_takeaways": _extract_or_default(sections, "practical_takeaways"),
        "limitations":         _extract_or_default(sections, "limitations_&_future_work",
                                                    "limitations_and_future_work",
                                                    "limitations"),
    })


# ── Dispatch table ────────────────────────────────────────────────────────────

_GENERATORS = {
    "SUMMARY.md":        generate_summary,
    "ARCHITECTURE.md":   generate_architecture,
    "IMPLEMENTATION.md": generate_implementation,
    "PLAN.md":           generate_plan,
    "RISKS.md":          generate_risks,
    "BENCHMARKS.md":     generate_benchmarks,
    "NEXT_STEPS.md":     generate_next_steps,
    # Input-type specific
    "ROOT_CAUSE.md":     generate_root_cause,
    "FIX_STEPS.md":      generate_fix_steps,
    "PREVENTION.md":     generate_prevention,
    "KEY_TAKEAWAYS.md":  generate_key_takeaways,
}


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_deliverables(
    topic: str,
    corpus_path: str | Path = "data/all_docs_cleaned.txt",
    output_dir: Optional[str | Path] = None,
    model: Optional[str] = None,
    run_id: Optional[str] = None,
    max_corpus_chars: int = 40_000,
    include_code: bool = True,
    input_type: Optional[str] = None,
) -> DeliverablePackage:
    """
    Generate a full research deliverable package.

    Args:
        topic:            Research topic string
        corpus_path:      Path to all_docs_cleaned.txt
        output_dir:       Where to write files (default: results/<slug>-<run_id>/)
        model:            LLM to use (None → auto-detect)
        run_id:           Unique run identifier (auto-generated if None)
        max_corpus_chars: Corpus window size
        include_code:     Whether to also run code_suggester
        input_type:       Override input type detection ('error_log', 'paper',
                          'codebase', 'website', 'text', or None for auto)

    Returns:
        DeliverablePackage with paths to all generated files
    """
    from research_deliverables.classify_topic import classify_topic
    from research_deliverables.classify_input import classify_input, input_deliverable_set
    from autoresearch.code_suggester import _sample_corpus_weighted

    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    # Auto run_id
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:40]
        run_id = f"{slug}-{ts}"

    # Output directory
    if output_dir is None:
        output_dir = corpus_path.parent.parent / "results" / run_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "CODE").mkdir(exist_ok=True)

    # Classify — input type takes priority over topic type
    corpus_sample = corpus_path.read_text(encoding="utf-8", errors="ignore")[:15_000]
    detected_input_type = classify_input(corpus_sample, hint=input_type)
    if detected_input_type != "text":
        deliverable_set = input_deliverable_set(detected_input_type, topic)
        print(f"[deliverables] Input type: {detected_input_type} (auto-detected)")
    else:
        deliverable_set = classify_topic(topic)
        print(f"[deliverables] Input type: text -> topic classification used")
    print(f"[deliverables] Topic type: {deliverable_set.research_type}")
    print(f"[deliverables] Deliverables: {', '.join(deliverable_set.deliverables)}")

    # Corpus sampling (external-first)
    corpus_extract, source_note = _sample_corpus_weighted(corpus_path, max_corpus_chars)
    corpus_text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    corpus_stats = {
        "chunks":    corpus_text.count("\n\n"),
        "chars":     f"{len(corpus_text):,}",
        "timestamp": timestamp,
    }
    print(f"[deliverables] Corpus: {len(corpus_extract):,} chars sampled  {source_note}")

    # Generate each deliverable
    files: dict[str, Path] = {}
    errors: dict[str, str] = {}

    for deliverable_name in deliverable_set.deliverables:
        gen_fn = _GENERATORS.get(deliverable_name)
        if gen_fn is None:
            continue
        try:
            content = gen_fn(
                topic=topic,
                corpus_extract=corpus_extract,
                source_note=source_note,
                deliverable_set=deliverable_set,
                run_id=run_id,
                corpus_stats=corpus_stats,
                model=model,
            )
            out_path = output_dir / deliverable_name
            out_path.write_text(content, encoding="utf-8")
            files[deliverable_name] = out_path
            print(f"[deliverables] ✅ {deliverable_name} → {out_path}")
        except Exception as exc:
            import traceback
            errors[deliverable_name] = str(exc)
            print(f"[deliverables] ❌ {deliverable_name}: {exc}", file=sys.stderr)
            traceback.print_exc()

    # Code suggestions (always generated into CODE/)
    if include_code:
        try:
            from autoresearch.code_suggester import generate_suggestions
            code_path = output_dir / "CODE" / "code_suggestions.md"
            print("[deliverables] Generating CODE/code_suggestions.md …")
            generate_suggestions(
                corpus_path=corpus_path,
                topic=topic,
                model=model,
                output_path=code_path,
            )
            files["CODE/code_suggestions.md"] = code_path
            print(f"[deliverables] ✅ CODE/code_suggestions.md → {code_path}")
        except Exception as exc:
            errors["CODE/code_suggestions.md"] = str(exc)
            print(f"[deliverables] ❌ CODE/code_suggestions.md: {exc}", file=sys.stderr)

    # Write metadata JSON
    meta = {
        "run_id":          run_id,
        "topic":           topic,
        "input_type":      detected_input_type,
        "research_type":   deliverable_set.research_type,
        "timestamp":       timestamp,
        "corpus_path":     str(corpus_path),
        "output_dir":      str(output_dir),
        "model":           model,
        "corpus_chars":    corpus_stats["chars"],
        "deliverables":    list(files.keys()),
        "errors":          errors,
        "source_note":     source_note,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\n[deliverables] Package complete → {output_dir}")
    if errors:
        print(f"[deliverables] ⚠️  {len(errors)} error(s): {', '.join(errors)}")

    return DeliverablePackage(
        run_id=run_id,
        topic=topic,
        research_type=deliverable_set.research_type,
        output_dir=output_dir,
        files=files,
        metadata=meta,
        errors=errors,
    )
