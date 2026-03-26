"""
autoresearch/code_suggester.py
------------------------------
Post-research code suggestion engine.

After the collector + prepare pipeline runs, this module:
  1. Reads the cleaned corpus (or a subset) to identify key concepts,
     APIs, patterns, and techniques the research surfaced.
  2. Sends them to an LLM with a "translate findings → code" prompt.
  3. Outputs ready-to-use code snippets + explanations as Markdown.

The output is saved to results/code_suggestions.md and is intentionally
practical: copy-paste examples, not theory.

Example: if your corpus is about Claude tool use / skills, the output will
contain concrete Python snippets using the Anthropic SDK showing how to
define tools, handle tool_use blocks, build multi-step agents, etc.

Usage (standalone):
    python -m autoresearch.code_suggester \
        --corpus data/all_docs_cleaned.txt \
        --topic "Claude tool use and skills" \
        --model gpt-4o-mini \
        --output results/code_suggestions.md

Usage (programmatic):
    from autoresearch.code_suggester import generate_suggestions
    md = generate_suggestions(corpus_path="data/all_docs_cleaned.txt")
    print(md)
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    """Load config.yaml from project root. Returns {} on any failure."""
    import logging
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        if cfg_path.exists():
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "[code_suggester] Failed to load config.yaml: %s — using defaults", exc
        )
    return {}

_CFG = _load_cfg()

# Max corpus chars to send to LLM — read from config.yaml, default 40k
# (Gemini 2.5 Flash Lite has 1M context; 40k = 3× improvement over old 12k)
CORPUS_WINDOW: int = _CFG.get("corpus", {}).get("window", 40_000)

# Fraction of CORPUS_WINDOW that should come from external sources
_EXTERNAL_FRACTION: float = _CFG.get("corpus", {}).get("external_fraction", 0.70)

# Number of code suggestion "blocks" to request
N_SUGGESTIONS = 3

SYSTEM_PROMPT = """\
You are an expert Principal Python Engineer. Your task is to translate research findings into immediately actionable, high-quality Python code.
You MUST output EXACTLY 3 major code implementation suggestions.

For each, strictly use this format:
## [Title]
**When to use:** [2-3 sentences explaining exactly when and why to use this pattern]

### Implementation Code
```python
# Fully self-contained, completely runnable Python code.
# NO placeholders, NO pseudocode. Use real SDK calls based on the research.
```

### Anti-Pattern to Avoid
[Show one explicit coding mistake developers make, and explain exactly why it fails]

Your response MUST ONLY contain these 3 sections in Markdown. Do not include introductory or concluding conversational text. Always finish your output completely.
"""

CORPUS_PROMPT_TEMPLATE = """\
Research topic: {topic}

{source_note}
--- Corpus extract ({n_chars} chars) ---
{corpus_extract}
--- End of extract ---

Based on the above, generate {n_suggestions} concrete Python code suggestions \
that a developer could use RIGHT NOW to apply these findings.

IMPORTANT:
- Prioritize insights from EXTERNAL sources (papers, GitHub repos, web articles) \
over patterns already in the project's own codebase.
- Generate NOVEL patterns that improve or extend existing code, not patterns \
that just describe what already exists.
- Focus on the most non-obvious, research-backed techniques.
"""


# ── LLM client ────────────────────────────────────────────────────────────────

def _call_llm(
    corpus_extract: str,
    topic: str,
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    n_suggestions: int = N_SUGGESTIONS,
    source_note: str = "",
) -> str:
    """Call LLM to generate code suggestions.
    "ollama:llama3.2"  → local Ollama (free, no key needed)
    """
    from autoresearch.llm_client import chat, best_available_model

    # Auto-detect model if none provided
    effective_model = model or best_available_model()

    user_content = CORPUS_PROMPT_TEMPLATE.format(
        topic=topic,
        n_chars=len(corpus_extract),
        corpus_extract=corpus_extract,
        n_suggestions=n_suggestions,
        source_note=source_note,
    )
    try:
        return chat(
            messages=[{"role": "user", "content": user_content}],
            model=effective_model,
            system=SYSTEM_PROMPT,
            max_tokens=min(1500 * n_suggestions, 8192),
            temperature=0.3,
            api_base=api_base,
            use_cache=False,
        )
    except Exception as exc:
        print(f"[code_suggester] LLM call failed: {exc}", file=sys.stderr)
        return _heuristic_suggestions(corpus_extract, topic)


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _extract_code_blocks(text: str, max_blocks: int = 6) -> list[tuple[str, str]]:
    """
    Pull out fenced code blocks from corpus text.
    Returns list of (lang, code) tuples, longest blocks first.
    """
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    blocks = [(m.group(1) or "python", m.group(2).strip())
              for m in pattern.finditer(text)
              if len(m.group(2).strip()) > 40]  # skip trivial one-liners
    # Sort by length so we surface the most complete examples
    blocks.sort(key=lambda b: len(b[1]), reverse=True)
    return blocks[:max_blocks]


def _extract_key_concepts(text: str, n: int = 8) -> list[str]:
    """
    Extract section headings and bold phrases as key concepts.
    Falls back to frequent meaningful bigrams if no headings found.
    """
    concepts: list[str] = []

    # Markdown headings: ## Foo, ### Bar
    for m in re.finditer(r"^#{1,3}\s+(.+)$", text, re.M):
        c = m.group(1).strip().rstrip(".")
        if 4 < len(c) < 80:
            concepts.append(c)

    # Bold phrases: **foo bar**
    for m in re.finditer(r"\*\*([^*]{4,60})\*\*", text):
        c = m.group(1).strip()
        if c not in concepts:
            concepts.append(c)

    if not concepts:
        # Fallback: most frequent meaningful bigrams
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        freq: dict[str, int] = {}
        for i in range(len(words) - 1):
            bg = f"{words[i]} {words[i+1]}"
            freq[bg] = freq.get(bg, 0) + 1
        concepts = [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]]

    return concepts[:n]


def _heuristic_suggestions(corpus_extract: str, topic: str) -> str:
    """
    Fallback when no LLM is available.

    Instead of an empty skeleton, this:
      1. Extracts real code blocks already present in the corpus.
      2. Lists the key concepts / headings surfaced by the research.
      3. Generates typed function stubs for the top detected patterns.
      4. Tells the user exactly what to run to get full LLM suggestions.
    """
    code_blocks = _extract_code_blocks(corpus_extract)
    concepts = _extract_key_concepts(corpus_extract)

    # Domain detection (for stub generation)
    text_lower = corpus_extract.lower()
    domain_hints = {
        "anthropic":     ("Anthropic Claude SDK", "import anthropic\nclient = anthropic.Anthropic()"),
        "tool_use":      ("Claude tool use",       "tools=[{\"name\": ..., \"input_schema\": {...}}]"),
        "lora":          ("LoRA fine-tuning",       "from peft import LoraConfig, get_peft_model"),
        "transformers":  ("HuggingFace Transformers","from transformers import AutoModelForCausalLM"),
        "scraper":       ("async web scraping",     "async with aiohttp.ClientSession() as session:"),
        "fastapi":       ("FastAPI",                "from fastapi import FastAPI\napp = FastAPI()"),
        "pydantic":      ("Pydantic models",        "from pydantic import BaseModel"),
        "streamlit":     ("Streamlit UI",           "import streamlit as st"),
        "sqlite":        ("SQLite storage",         "import sqlite3\nconn = sqlite3.connect('db.sqlite3')"),
        "skill":         ("Claude Skill pattern",   "# SKILL.md trigger + Python implementation"),
    }
    detected = [(label, snippet)
                for kw, (label, snippet) in domain_hints.items()
                if kw in text_lower]

    parts: list[str] = []
    parts.append("# Code Suggestions (Heuristic — No LLM Available)\n")
    parts.append(
        "> ⚠️  Ollama / API key not available. "
        "The snippets below are extracted **directly from your corpus** "
        "and augmented with typed stubs. "
        "Re-run with an LLM for fully synthesised, topic-specific suggestions.\n"
    )

    # ── Section 1: Key concepts found ─────────────────────────────────────────
    if concepts:
        parts.append("## Key Concepts Found in Corpus\n")
        parts.append("The research surfaces these topics (in order of prominence):\n")
        for i, c in enumerate(concepts, 1):
            parts.append(f"{i}. {c}")
        parts.append("")

    # ── Section 2: Real code from the corpus ──────────────────────────────────
    if code_blocks:
        parts.append(f"## Code Extracted from Corpus ({len(code_blocks)} blocks)\n")
        parts.append(
            "These snippets appeared in your research documents. "
            "They are the most complete examples the corpus contains:\n"
        )
        for i, (lang, code) in enumerate(code_blocks, 1):
            parts.append(f"### Example {i}\n")
            parts.append(f"```{lang or 'python'}")
            parts.append(code)
            parts.append("```\n")
    else:
        parts.append("## Code Extracted from Corpus\n")
        parts.append("_No fenced code blocks found in the corpus excerpt._\n")

    # ── Section 3: Typed stubs for detected patterns ──────────────────────────
    if detected:
        parts.append("## Typed Stubs for Detected Patterns\n")
        parts.append(
            "These stubs are generated from keywords found in the corpus. "
            "Fill in the bodies based on the concepts listed above:\n"
        )
        for label, import_line in detected[:4]:
            fn_name = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            parts.append(f"### {label}\n")
            parts.append("```python")
            parts.append(f"# Pattern: {label}")
            parts.append(import_line)
            parts.append("")
            parts.append(f"def {fn_name}():")
            parts.append(f'    """TODO: implement {label} pattern."""')
            parts.append("    raise NotImplementedError")
            parts.append("```\n")

    # ── Section 4: How to get full LLM suggestions ────────────────────────────
    parts.append("## Get Full Suggestions (requires LLM)\n")
    parts.append("```bash")
    parts.append("# Option A — local Ollama (free)")
    parts.append("ollama pull llama3.2")
    parts.append("python -m autoresearch.code_suggester \\")
    parts.append("    --corpus data/all_docs_cleaned.txt \\")
    if topic:
        parts.append(f'    --topic "{topic}" \\')
    parts.append("    --model ollama:llama3.2")
    parts.append("")
    parts.append("# Option B — Anthropic (best quality)")
    parts.append("ANTHROPIC_API_KEY=sk-ant-... python -m autoresearch.code_suggester \\")
    parts.append("    --corpus data/all_docs_cleaned.txt \\")
    if topic:
        parts.append(f'    --topic "{topic}" \\')
    parts.append("    --model claude-3-5-haiku-20241022")
    parts.append("```")

    return "\n".join(parts)


# ── Topic detection ───────────────────────────────────────────────────────────

def _detect_topic(corpus: str, program_md_path: Optional[Path] = None) -> str:
    """
    Auto-detect the research topic. Tries (in order):
      1. program.md title line
      2. Most frequent meaningful n-grams in corpus
    """
    if program_md_path and program_md_path.exists():
        text = program_md_path.read_text(encoding="utf-8")
        # Look for '# Title' or 'Topic: ...' lines
        m = re.search(r"^(?:#\s*|topic:\s*)(.+)$", text, re.I | re.M)
        if m:
            return m.group(1).strip()

    # Fallback: count most common 2-grams
    words = re.findall(r"\b[a-z]{4,}\b", corpus.lower())
    freq: dict[str, int] = {}
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        freq[bigram] = freq.get(bigram, 0) + 1
    if freq:
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
        return ", ".join(t[0] for t in top)
    return "research findings"


# ── Corpus sampling ───────────────────────────────────────────────────────────

def _sample_corpus(corpus: str, max_chars: int = CORPUS_WINDOW) -> str:
    """
    Select the most representative portion of a corpus for the LLM prompt.
    Strategy: beginning + middle + end spread to maximise coverage.
    """
    if len(corpus) <= max_chars:
        return corpus

    third = max_chars // 3
    beginning = corpus[:third]
    mid_start = len(corpus) // 2 - third // 2
    middle = corpus[mid_start: mid_start + third]
    end = corpus[-third:]
    return f"{beginning}\n\n[...middle sample...]\n\n{middle}\n\n[...end sample...]\n\n{end}"


def _sample_corpus_weighted(
    corpus_path: Path,
    max_chars: int = CORPUS_WINDOW,
    external_fraction: float = _EXTERNAL_FRACTION,
) -> tuple[str, str]:
    """
    Build a corpus sample that prioritises external research sources.

    Looks for data/external_docs.txt (written by analyzer.py) alongside
    the main corpus. If found, fills `external_fraction` of the window
    from it and the remainder from the main corpus (which may include
    internal docs).

    Returns (sampled_text, source_note) where source_note is a one-line
    header explaining the mix — included in the LLM prompt for transparency.
    """
    external_path = corpus_path.parent / "external_docs.txt"
    main_corpus = corpus_path.read_text(encoding="utf-8", errors="ignore")

    if not external_path.exists():
        # No split available — use full corpus with a note
        sampled = _sample_corpus(main_corpus, max_chars)
        note = "(Using combined corpus — run analyzer.py to enable external-first sampling)"
        return sampled, note

    external_corpus = external_path.read_text(encoding="utf-8", errors="ignore")
    ext_budget = int(max_chars * external_fraction)
    int_budget = max_chars - ext_budget

    ext_sample = _sample_corpus(external_corpus, ext_budget)
    int_sample = _sample_corpus(main_corpus, int_budget)

    # External content first in the prompt
    combined = ext_sample
    if int_sample.strip():
        combined += f"\n\n[--- Additional context from project codebase ---]\n\n{int_sample}"

    ext_pct = int(external_fraction * 100)
    int_pct = 100 - ext_pct
    note = (
        f"(Corpus mix: ~{ext_pct}% external research sources, "
        f"~{int_pct}% project codebase context)"
    )
    return combined, note


# ── Markdown wrapper ──────────────────────────────────────────────────────────

def _wrap_in_report(suggestions_md: str, topic: str, corpus_stats: dict) -> str:
    chars = corpus_stats.get('chars', 0)
    chars_str = f"{chars:,}" if isinstance(chars, int) else str(chars)
    header = textwrap.dedent(f"""\
        # Code Suggestions from Research

        **Topic:** {topic}
        **Corpus:** {corpus_stats.get('chunks', '?')} chunks, {chars_str} chars
        **Generated:** {corpus_stats.get('timestamp', '')}

        These suggestions were generated by analysing your research corpus and
        translating the key patterns into ready-to-use Python code.

        ---

    """)
    footer = textwrap.dedent("""

        ---

        ## How to use these suggestions

        1. Copy the snippet most relevant to your use case.
        2. Install any dependencies shown in the imports.
        3. Replace placeholder values (`YOUR_API_KEY`, `your_model`, etc.)
           with your actual config.
        4. Run `python -m autoresearch.code_suggester --help` to regenerate
           suggestions after collecting more documents.

        > Re-run after each research iteration to get updated suggestions
        > as your corpus grows.
    """)
    return header + suggestions_md + footer


# ── Main pipeline ─────────────────────────────────────────────────────────────

def generate_suggestions(
    corpus_path: str | Path = "data/all_docs_cleaned.txt",
    topic: str = "",
    model: Optional[str] = None,   # None → auto-detect best available model
    api_base: Optional[str] = None,
    output_path: str | Path = "results/code_suggestions.md",
    program_md: str | Path = "templates/program.md",
    n_suggestions: int = N_SUGGESTIONS,
    max_corpus_chars: int = CORPUS_WINDOW,
) -> str:
    """
    Full code suggestion pipeline.

    Args:
        corpus_path:      cleaned corpus from analyzer.py
        topic:            research topic (auto-detected if empty)
        model:            LLM for suggestion generation
        api_base:         override OpenAI API base
        output_path:      where to write the Markdown report
        program_md:       used for topic auto-detection
        n_suggestions:    how many code snippets to generate
        max_corpus_chars: max chars of corpus to send to LLM

    Returns:
        Full Markdown string (also written to output_path)
    """
    from datetime import datetime, timezone

    corpus_path = Path(corpus_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    corpus = corpus_path.read_text(encoding="utf-8")
    print(f"[code_suggester] Loaded corpus: {len(corpus):,} chars")

    # Topic detection
    if not topic:
        topic = _detect_topic(corpus, Path(program_md))
    print(f"[code_suggester] Topic: {topic}")

    # External-weighted sampling (70% external research, 30% internal codebase)
    extract, source_note = _sample_corpus_weighted(corpus_path, max_corpus_chars)
    print(f"[code_suggester] Corpus sample: {len(extract):,} chars  {source_note}")

    # Resolve model: explicit arg → best available → heuristic
    from autoresearch.llm_client import best_available_model
    effective_model = model or best_available_model()
    print(f"[code_suggester] Using model: {effective_model}")

    # Generate suggestions via LLM
    print(f"[code_suggester] Generating {n_suggestions} suggestions with {effective_model}…")
    suggestions_md = _call_llm(
        corpus_extract=extract,
        topic=topic,
        model=effective_model,
        api_base=api_base,
        n_suggestions=n_suggestions,
        source_note=source_note,
    )

    # Wrap in report
    stats = {
        "chunks": corpus.count("\n\n"),
        "chars": len(corpus),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    full_report = _wrap_in_report(suggestions_md, topic, stats)

    # Save
    output_path.write_text(full_report, encoding="utf-8")
    print(f"[code_suggester] ✅ Saved to {output_path}")
    return full_report


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    import argparse

    parser = argparse.ArgumentParser(description="Generate code suggestions from research corpus")
    parser.add_argument("--corpus", default="data/all_docs_cleaned.txt")
    parser.add_argument("--topic", default="", help="Override topic auto-detection")
    parser.add_argument("--model", default=None,
                        help="LLM model: gpt-4o-mini, claude-* etc. (defaults to best available)")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--output", default="results/code_suggestions.md")
    parser.add_argument("--program", default="templates/program.md")
    parser.add_argument("--n-suggestions", type=int, default=N_SUGGESTIONS)
    args = parser.parse_args()

    md = generate_suggestions(
        corpus_path=args.corpus,
        topic=args.topic,
        model=args.model,
        api_base=args.api_base,
        output_path=args.output,
        program_md=args.program,
        n_suggestions=args.n_suggestions,
    )
    # Print first 1000 chars as preview
    print("\n" + "─" * 60)
    print("Preview (first 1000 chars):")
    print("─" * 60)
    print(md[:1000])
