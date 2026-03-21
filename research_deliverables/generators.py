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
              max_tokens: int = 2048) -> str:
    """Call the LLM. Returns plain string response."""
    try:
        from autoresearch.llm_client import chat, best_available_model
        effective_model = model or best_available_model()
        return chat(
            messages=[{"role": "user", "content": user}],
            model=effective_model,
            system=system,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return f"[LLM unavailable: {exc}]\n\n_Run with a configured LLM to generate this section._"


# ── Structured extraction ─────────────────────────────────────────────────────

_SECTION_RE = re.compile(
    r"##\s+(?P<heading>[^\n]+)\n(?P<body>.*?)(?=\n##\s|\Z)",
    re.DOTALL,
)

def _extract_sections(text: str) -> dict[str, str]:
    """Parse ## heading / body pairs from LLM output into a dict."""
    sections: dict[str, str] = {}
    for m in _SECTION_RE.finditer(text):
        key = m.group("heading").strip().lower().replace(" ", "_")
        sections[key] = m.group("body").strip()
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

_SYSTEM_PROMPTS = {
    "SUMMARY.md":        _SUMMARY_SYSTEM,
    "ARCHITECTURE.md":   _ARCHITECTURE_SYSTEM,
    "IMPLEMENTATION.md": _IMPLEMENTATION_SYSTEM,
    "PLAN.md":           _PLAN_SYSTEM,
    "RISKS.md":          _RISKS_SYSTEM,
    "BENCHMARKS.md":     _BENCHMARKS_SYSTEM,
    "NEXT_STEPS.md":     _NEXT_STEPS_SYSTEM,
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


# ── Dispatch table ────────────────────────────────────────────────────────────

_GENERATORS = {
    "SUMMARY.md":        generate_summary,
    "ARCHITECTURE.md":   generate_architecture,
    "IMPLEMENTATION.md": generate_implementation,
    "PLAN.md":           generate_plan,
    "RISKS.md":          generate_risks,
    "BENCHMARKS.md":     generate_benchmarks,
    "NEXT_STEPS.md":     generate_next_steps,
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

    Returns:
        DeliverablePackage with paths to all generated files
    """
    from research_deliverables.classify_topic import classify_topic
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

    # Classify
    deliverable_set = classify_topic(topic)
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
