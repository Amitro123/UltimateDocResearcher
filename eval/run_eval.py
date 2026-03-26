"""
eval/run_eval.py
----------------
Standardized 5-criteria LLM evaluation framework for UltimateDocResearcher.

Scores any research output (code_suggestions.md, rules files, skill
descriptions) against the criteria defined in eval/eval_spec.yaml using
an LLM as judge.

Usage:
    python -m eval.run_eval \\
        --input results/code_suggestions.md \\
        --judge ollama:llama3.2 \\
        --threshold 3.5 \\
        --output results/eval-report.json

    # With Anthropic
    python -m eval.run_eval \\
        --input results/code_suggestions.md \\
        --judge claude-3-5-haiku-20241022 \\
        --output results/eval-report.json

    # Batch — score multiple files
    python -m eval.run_eval \\
        --input results/*.md \\
        --judge ollama:llama3.2
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── YAML loading ──────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    """Load eval_spec.yaml using PyYAML (required, listed in requirements.txt)."""
    import yaml
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Load spec ─────────────────────────────────────────────────────────────────

SPEC_PATH = Path(__file__).parent / "eval_spec.yaml"


def load_spec(spec_path: Path = SPEC_PATH) -> list[dict]:
    """Load and return the list of criteria from eval_spec.yaml."""
    data = _load_yaml(spec_path)
    criteria = data.get("criteria", [])
    if not criteria:
        raise ValueError(f"No criteria found in {spec_path}")
    return criteria


# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are an expert evaluator assessing the quality of AI-generated research outputs.
You will be given a document and a specific evaluation question.
Respond with ONLY this format (no other text):

Score: <integer 1-5>
Reasoning: <one concise sentence explaining the score>

Scoring scale: 1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent.
Be strict. Reserve 5 for truly exceptional outputs.
"""


def _judge_criterion(
    document: str,
    criterion: dict,
    judge_model: str,
    api_base: Optional[str] = None,
) -> tuple[int, str]:
    """
    Ask the judge LLM to score one criterion.
    Returns (score: int, reasoning: str).
    Falls back to heuristic score on any failure.
    """
    question = criterion.get("question", "").strip()
    name = criterion.get("name", "unknown")

    user_content = (
        f"Evaluation question: {question}\n\n"
        f"--- Document to evaluate ---\n"
        f"{document[:8000]}\n"
        f"--- End of document ---"
    )

    try:
        from autoresearch.llm_client import chat
        raw = chat(
            messages=[{"role": "user", "content": user_content}],
            model=judge_model,
            system=JUDGE_SYSTEM,
            max_tokens=256,
            temperature=0.0,
            api_base=api_base,
        )
        return _parse_score(raw)
    except Exception as exc:
        short = str(exc).splitlines()[0][:80]
        print(f"[eval] '{name}': LLM unavailable ({short}) — heuristic fallback", file=sys.stderr)
        return _heuristic_score(document, criterion), "heuristic fallback (no LLM)"


def _parse_score(text: str) -> tuple[int, str]:
    """Parse 'Score: N\\nReasoning: ...' from judge output."""
    score_m = re.search(r"Score:\s*([1-5])", text, re.I)
    reason_m = re.search(r"Reasoning:\s*(.+)", text, re.I | re.DOTALL)

    score = int(score_m.group(1)) if score_m else 3
    reasoning = reason_m.group(1).strip()[:500] if reason_m else text[:300]
    return score, reasoning


def _heuristic_score(document: str, criterion: dict) -> int:
    """
    Fallback when no LLM is available.
    Uses keyword signals relevant to each criterion.

    Per-criterion ceilings prevent generic signals from inflating scores:
    - clarity/freshness cap at 3 because their keywords are too common or unreliable
      as proxies for quality (e.g. "example" appears in every doc; year strings in
      auto-generated text don't guarantee up-to-date content).
    """
    name = criterion.get("name", "")
    doc_lower = document.lower()

    signals = {
        # Narrowed: removed "example" and "means" — ubiquitous in any code doc,
        # making clarity score 4 even for low-quality outputs.
        "clarity": ["e.g.", "for instance", "specifically"],
        "completeness": ["also", "additionally", "furthermore", "covers", "includes"],
        "actionability": ["```python", "import ", "def ", "pip install", "copy"],
        "freshness": ["2025", "2026", "latest", "updated", "modern"],
        "anti_patterns": ["don't", "avoid", "gotcha", "common mistake", "never", "instead"],
    }

    # Per-criterion max: clarity and freshness are capped at 3 to reduce
    # optimism bias observed when no LLM is available (walkthrough finding).
    ceilings = {
        "clarity": 3,
        "completeness": 4,
        "actionability": 4,
        "freshness": 3,
        "anti_patterns": 4,
    }

    hits = sum(1 for kw in signals.get(name, []) if kw in doc_lower)
    ceiling = ceilings.get(name, 4)
    # Map hit count → score: 0→2, 1→3, 2+→4, capped by ceiling
    return min(ceiling, max(2, hits + 2))


# ── Weighted scoring ──────────────────────────────────────────────────────────

def compute_weighted_score(
    scores: dict[str, int], criteria: list[dict]
) -> float:
    """Compute weighted average across all criteria."""
    total_weight = sum(c.get("weight", 1.0) for c in criteria)
    weighted_sum = sum(
        scores.get(c["name"], 3) * c.get("weight", 1.0)
        for c in criteria
    )
    return round(weighted_sum / total_weight, 3) if total_weight else 0.0


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate(
    input_path: str | Path,
    judge_model: str = "ollama:llama3.2",
    threshold: float = 3.5,
    output_path: Optional[str | Path] = None,
    spec_path: Path = SPEC_PATH,
    api_base: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a research output document against the 5-criteria spec.

    Args:
        input_path:   path to the document to evaluate (Markdown or text)
        judge_model:  LLM judge (ollama:llama3.2, claude-*, gpt-4o-mini)
        threshold:    pass/fail threshold for weighted_avg (default 3.5)
        output_path:  where to save JSON report (None → results/eval-report.json)
        spec_path:    override path to eval_spec.yaml
        api_base:     override API base URL for LLM
        verbose:      print progress and summary to stdout

    Returns:
        report dict with summary + per-criterion results
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    document = input_path.read_text(encoding="utf-8")
    criteria = load_spec(spec_path)

    if verbose:
        print(f"\n[eval] Evaluating: {input_path.name}")
        print(f"[eval] Judge: {judge_model} | Threshold: {threshold}")
        print(f"[eval] {len(criteria)} criteria loaded from {spec_path.name}")
        print(f"[eval] Document: {len(document):,} chars\n")

    # Score each criterion
    criterion_results = []
    scores: dict[str, int] = {}

    for c in criteria:
        name = c["name"]
        weight = c.get("weight", 1.0)

        if verbose:
            print(f"  Scoring '{name}' (weight={weight})…", end=" ", flush=True)

        score, reasoning = _judge_criterion(document, c, judge_model, api_base)
        scores[name] = score
        weighted = round(score * weight, 3)

        criterion_results.append({
            "name": name,
            "score": score,
            "weight": weight,
            "weighted_score": weighted,
            "reasoning": reasoning,
        })

        if verbose:
            print(f"{score}/5")

    # Compute final score
    weighted_avg = compute_weighted_score(scores, criteria)
    passed = weighted_avg >= threshold

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge_model,
        "input_file": str(input_path),
        "threshold_used": threshold,
        "weighted_avg": weighted_avg,
        "passed": passed,
        "criterion_scores": scores,
    }

    report = {
        "summary": summary,
        "criteria": criterion_results,
    }

    # Save report
    if output_path is None:
        out_dir = input_path.parent / "eval-results"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"eval-report-{input_path.stem}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if verbose:
        _print_summary(summary, criterion_results)
        print(f"\n[eval] Report saved → {output_path}")

    return report


# ── Pretty-print ──────────────────────────────────────────────────────────────

def _print_summary(summary: dict, criterion_results: list[dict]) -> None:
    sep = "─" * 56
    passed = summary["passed"]
    verdict = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{sep}")
    print(f"  📊  Eval Report  ({verdict})")
    print(sep)
    print(f"  Judge          : {summary['judge_model']}")
    print(f"  Input          : {Path(summary['input_file']).name}")
    print(f"  Threshold      : {summary['threshold_used']}")
    print(f"  Weighted score : {summary['weighted_avg']:.2f} / 5.00")
    print()
    for r in criterion_results:
        bar = "█" * r["score"] + "░" * (5 - r["score"])
        print(f"  {r['name']:14s} [{bar}] {r['score']}/5  (×{r['weight']})")
    print()
    # Print shortest reasoning per criterion
    for r in criterion_results:
        short = r["reasoning"][:80].replace("\n", " ")
        print(f"  {r['name']:14s} → {short}")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    import argparse
    import glob as _glob

    parser = argparse.ArgumentParser(
        description="Score research outputs against 5-criteria eval spec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m eval.run_eval --input results/code_suggestions.md --judge ollama:llama3.2
              python -m eval.run_eval --input results/*.md --judge claude-3-5-haiku-20241022
              python -m eval.run_eval --input results/code_suggestions.md --threshold 4.0
        """),
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Path(s) to document(s) to evaluate (supports glob patterns)",
    )
    parser.add_argument(
        "--judge", default="ollama:llama3.2",
        help="LLM judge: ollama:llama3.2, claude-3-5-haiku-20241022, gpt-4o-mini",
    )
    parser.add_argument(
        "--threshold", type=float, default=3.5,
        help="Pass/fail threshold for weighted average (default: 3.5)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: eval-results/eval-report-<stem>.json)",
    )
    parser.add_argument(
        "--spec", default=str(SPEC_PATH),
        help="Path to eval_spec.yaml (default: eval/eval_spec.yaml)",
    )
    parser.add_argument(
        "--api-base", default=None,
        help="Override API base URL for LLM calls",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    # Expand glob patterns
    input_files: list[Path] = []
    for pattern in args.input:
        expanded = _glob.glob(pattern)
        if expanded:
            input_files.extend(Path(p) for p in expanded)
        else:
            input_files.append(Path(pattern))

    if not input_files:
        print("[eval] No input files found.", file=sys.stderr)
        sys.exit(1)

    spec_path = Path(args.spec)
    all_passed = True

    for input_file in input_files:
        # For multiple files, auto-generate per-file output paths
        out = args.output if len(input_files) == 1 else None

        try:
            report = evaluate(
                input_path=input_file,
                judge_model=args.judge,
                threshold=args.threshold,
                output_path=out,
                spec_path=spec_path,
                api_base=args.api_base,
                verbose=not args.quiet,
            )
            if not report.get("summary", {}).get("passed", False):
                all_passed = False
        except FileNotFoundError as e:
            print(f"[eval] ❌ {e}", file=sys.stderr)
            all_passed = False

    # Exit with non-zero code if any file failed — useful in CI
    sys.exit(0 if all_passed else 1)
