"""
autoresearch/eval.py
--------------------
LLM-as-a-Judge evaluator for the autoresearch pipeline.

Loads val.jsonl (Q&A pairs), optionally generates model predictions,
then uses a judge LLM to score each answer on three axes:
  • accuracy   — is the answer factually correct?
  • relevance  — does it answer the question that was asked?
  • completeness — does it cover the key points from the reference?

Each axis is scored 1-5. Results are saved to results/eval_report.json
and a one-line summary is appended to results/results.tsv.

Usage (standalone):
    python -m autoresearch.eval \
        --val-path data/val.jsonl \
        --judge-model gpt-4o-mini \
        --max-samples 50

Usage (programmatic):
    from autoresearch.eval import run_eval
    report = run_eval(val_path="data/val.jsonl", judge_model="gpt-4o-mini")
    print(report["summary"])
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class JudgeScores:
    accuracy: int       # 1-5
    relevance: int      # 1-5
    completeness: int   # 1-5
    reasoning: str      # judge's explanation
    overall: float      # weighted average

    @classmethod
    def from_parse(cls, text: str) -> "JudgeScores":
        """Parse judge LLM output into structured scores."""
        def _extract(label: str) -> int:
            m = re.search(rf"{label}[:\s]+([1-5])", text, re.I)
            if m:
                return int(m.group(1))
            print(
                f"[eval] ⚠️  Could not parse '{label}' score from judge output — "
                f"defaulting to 3. Judge response was: {text[:200]!r}",
                file=sys.stderr,
            )
            return 3  # default middle score

        acc = _extract("accuracy")
        rel = _extract("relevance")
        comp = _extract("completeness")
        # Extract reasoning block (use .* so empty trailing text still matches)
        reasoning_m = re.search(r"(?:reasoning|explanation)[:\s]*(.*)", text, re.I | re.DOTALL)
        if reasoning_m:
            reasoning = reasoning_m.group(1).strip()[:500]
        else:
            reasoning = text[:300]
        if not reasoning:
            reasoning = f"(judge provided no explanation — raw: {text[:200]})"
        overall = round((acc * 0.4 + rel * 0.35 + comp * 0.25), 2)
        return cls(acc, rel, comp, reasoning, overall)

    def is_passing(self, threshold: float = 3.0) -> bool:
        return self.overall >= threshold


@dataclass
class SampleResult:
    question: str
    reference_answer: str
    model_answer: str
    scores: JudgeScores
    passed: bool


# ── Judge prompt ───────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are an expert evaluator assessing the quality of AI-generated answers.
Score the provided answer against the reference on three axes (1=very poor, 5=excellent):

  Accuracy     – factual correctness compared to the reference
  Relevance    – how directly it answers the question
  Completeness – coverage of key points in the reference

Reply in this exact format (you MUST write at least one full sentence after "Reasoning:"):
  Accuracy: <1-5>
  Relevance: <1-5>
  Completeness: <1-5>
  Reasoning: <one short paragraph explaining your scores>
"""

def _judge_prompt(question: str, reference: str, model_answer: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Reference answer: {reference[:800]}\n\n"
        f"Model answer to evaluate: {model_answer[:800]}"
    )


# ── LLM client ────────────────────────────────────────────────────────────────

def _call_judge(
    question: str,
    reference: str,
    model_answer: str,
    judge_model: Optional[str] = None,
    api_base: Optional[str] = None,
) -> JudgeScores:
    """
    Call judge LLM via the unified llm_client.
    """
    from autoresearch.llm_client import chat, best_available_model
    
    # Auto-detect judge model
    effective_judge = judge_model or best_available_model()

    try:
        raw = chat(
            messages=[{"role": "user", "content": _judge_prompt(question, reference, model_answer)}],
            model=effective_judge,
            system=JUDGE_SYSTEM,
            max_tokens=512,
            temperature=0.0,
            api_base=api_base,
        )
        return JudgeScores.from_parse(raw)
    except Exception as exc:
        short = str(exc).splitlines()[0][:80]
        print(f"[eval] Judge unavailable ({short}) — heuristic fallback activated", file=sys.stderr)
        return _heuristic_score(reference, model_answer)


def _heuristic_score(reference: str, model_answer: str) -> JudgeScores:
    """
    Fallback heuristic when no LLM is available.
    Uses word-overlap as a proxy for quality.
    """
    ref_words = set(reference.lower().split())
    ans_words = set(model_answer.lower().split())
    if not ref_words:
        overlap = 0.0
    else:
        overlap = len(ref_words & ans_words) / len(ref_words)

    # Map overlap [0,1] → score [1,5]
    score = int(1 + overlap * 4)
    score = max(1, min(5, score))
    return JudgeScores(
        accuracy=score,
        relevance=score,
        completeness=score,
        reasoning=f"Heuristic: word-overlap={overlap:.2f} (no LLM judge available)",
        overall=float(score),
    )


# ── Model inference (optional) ────────────────────────────────────────────────

def _generate_model_answer(
    question: str,
    system_prompt: str = "You are a knowledgeable research assistant.",
    model_path: Optional[str] = None,
) -> str:
    """
    Generate an answer from the fine-tuned model (if available).
    Falls back to returning 'N/A' gracefully so eval still works
    against reference answers directly.
    """
    if model_path is None:
        # No model: evaluate reference against itself to sanity-check scoring
        return "[no model — evaluating reference answer quality]"

    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "text-generation",
            model=model_path,
            max_new_tokens=256,
            temperature=0.0,
        )
        prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"
        out = pipe(prompt)[0]["generated_text"]
        # Strip the prompt prefix
        answer = out[len(prompt):].strip()
        return answer or "[empty model output]"
    except Exception as exc:
        print(f"[eval] Model inference failed: {exc}", file=sys.stderr)
        return "[model inference unavailable]"


# ── Main eval pipeline ────────────────────────────────────────────────────────

def run_eval(
    val_path: str | Path = "data/val.jsonl",
    model_path: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    judge_api_base: Optional[str] = None,
    max_samples: int = 50,
    pass_threshold: float = 3.0,
    output_dir: str | Path = "results/",
    iteration: int = 0,
    topic: str = "",
) -> dict:
    """
    Full eval pipeline.

    Args:
        val_path:       path to val.jsonl from prepare.py
        model_path:     optional fine-tuned model dir (None → evaluates reference quality)
        judge_model:    LLM to use as judge (OpenAI model name or 'claude-*')
        judge_api_base: override API base URL (for local models serving OpenAI API)
        max_samples:    cap on how many val samples to evaluate
        pass_threshold: overall score >= this → sample "passes"
        output_dir:     where to write eval_report.json
        iteration:      training iteration number (for tracking)
        topic:          research topic label

    Returns:
        report dict with summary + per-sample results
    """
    # Ensure judge_model is always a string (CLI passes None when flag is omitted)
    if judge_model is None:
        from autoresearch.llm_client import best_available_model
        judge_model = best_available_model()

    val_path = Path(val_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not val_path.exists():
        raise FileNotFoundError(f"val.jsonl not found: {val_path}")

    # Load val samples
    samples = []
    with val_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    samples = samples[:max_samples]
    print(f"[eval] Evaluating {len(samples)} samples with judge={judge_model}")

    results: list[SampleResult] = []

    try:
        for i, sample in enumerate(samples):
            msgs = sample.get("messages", [])
            # Extract Q and A from chat format
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            ref_answer = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            system_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")

            if not user_msg or not ref_answer:
                continue

            # Generate model prediction (or use placeholder)
            model_answer = _generate_model_answer(user_msg, system_msg, model_path)

            # Judge scoring
            scores = _call_judge(
                question=user_msg,
                reference=ref_answer,
                model_answer=model_answer,
                judge_model=judge_model,
                api_base=judge_api_base,
            )

            results.append(SampleResult(
                question=user_msg,
                reference_answer=ref_answer,
                model_answer=model_answer,
                scores=scores,
                passed=scores.is_passing(pass_threshold),
            ))

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(samples)}] avg overall so far: "
                      f"{statistics.mean(r.scores.overall for r in results):.2f}", flush=True)
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n[eval] Interrupted by user. Saving partial report...")
        else:
            print(f"\n[eval] Error during evaluation: {e}. Saving partial report...", file=sys.stderr)

    if not results:
        print("[eval] ⚠️  No results produced — check val.jsonl format or LLM availability")
        return {}

    def _get_score(r: SampleResult, attr: str, default: float = 3.0) -> float:
        """Safely extract a score attribute from JudgeScores or a plain dict fallback."""
        scores = r.scores
        if isinstance(scores, dict):
            return float(scores.get(attr, default))
        return float(getattr(scores, attr, default))

    # Aggregate
    all_overall = [_get_score(r, "overall") for r in results]
    all_acc = [_get_score(r, "accuracy") for r in results]
    all_rel = [_get_score(r, "relevance") for r in results]
    all_comp = [_get_score(r, "completeness") for r in results]
    pass_rate = sum(r.passed for r in results) / len(results)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "topic": topic,
        "judge_model": judge_model,
        "n_samples": len(results),
        "pass_threshold": pass_threshold,
        "pass_rate": round(pass_rate, 3),
        "avg_overall": round(statistics.mean(all_overall), 3),
        "avg_accuracy": round(statistics.mean(all_acc), 3),
        "avg_relevance": round(statistics.mean(all_rel), 3),
        "avg_completeness": round(statistics.mean(all_comp), 3),
        "stdev_overall": round(statistics.stdev(all_overall), 3) if len(all_overall) > 1 else 0.0,
        "worst_samples": _worst_samples(results, n=3),
        "best_samples": _best_samples(results, n=3),
    }

    report = {
        "summary": summary,
        "samples": [
            {
                "question": r.question[:200],
                "reference": r.reference_answer[:300],
                "model_answer": r.model_answer[:300],
                "scores": asdict(r.scores),
                "passed": r.passed,
            }
            for r in results
        ],
    }

    # Write report
    report_path = output_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Pretty-print summary
    _print_summary(summary)

    return report


def _worst_samples(results: list[SampleResult], n: int = 3) -> list[dict]:
    def _safe_overall(r: SampleResult) -> float:
        s = r.scores
        return float(s.get("overall", 3.0) if isinstance(s, dict) else getattr(s, "overall", 3.0))

    def _safe_reasoning(r: SampleResult) -> str:
        s = r.scores
        val = s.get("reasoning", "") if isinstance(s, dict) else getattr(s, "reasoning", "")
        return str(val)[:200]

    sorted_r = sorted(results, key=_safe_overall)
    return [
        {"question": r.question[:150], "overall": _safe_overall(r), "reasoning": _safe_reasoning(r)}
        for r in sorted_r[:n]
    ]


def _best_samples(results: list[SampleResult], n: int = 3) -> list[dict]:
    def _safe_overall(r: SampleResult) -> float:
        s = r.scores
        return float(s.get("overall", 3.0) if isinstance(s, dict) else getattr(s, "overall", 3.0))

    def _safe_reasoning(r: SampleResult) -> str:
        s = r.scores
        val = s.get("reasoning", "") if isinstance(s, dict) else getattr(s, "reasoning", "")
        return str(val)[:200]

    sorted_r = sorted(results, key=_safe_overall, reverse=True)
    return [
        {"question": r.question[:150], "overall": _safe_overall(r), "reasoning": _safe_reasoning(r)}
        for r in sorted_r[:n]
    ]


def _print_summary(summary: dict) -> None:
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  📊  Eval Summary  (judge: {summary['judge_model']})")
    print(sep)
    print(f"  Samples evaluated : {summary['n_samples']}")
    print(f"  Pass rate         : {summary['pass_rate']*100:.1f}%  (threshold ≥ {summary['pass_threshold']})")
    print(f"  Avg overall score : {summary['avg_overall']:.2f} / 5")
    print(f"  ├─ Accuracy       : {summary['avg_accuracy']:.2f}")
    print(f"  ├─ Relevance      : {summary['avg_relevance']:.2f}")
    print(f"  └─ Completeness   : {summary['avg_completeness']:.2f}")
    if summary.get("worst_samples"):
        print(f"\n  ⚠️  Lowest-scoring questions:")
        for s in summary["worst_samples"]:
            print(f"    [{s['overall']:.1f}] {s['question'][:80]}…")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluator")
    parser.add_argument("--val-path", default="data/val.jsonl")
    parser.add_argument("--model-path", default=None, help="Path to fine-tuned model (optional)")
    parser.add_argument("--judge-model", default=None,
                        help="Judge LLM: gpt-4o-mini, claude-* etc. (defaults to best available)")
    parser.add_argument("--judge-api-base", default=None, help="Override OpenAI API base URL")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--pass-threshold", type=float, default=3.0)
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--topic", default="")
    args = parser.parse_args()

    try:
        run_eval(
            val_path=args.val_path,
            model_path=args.model_path,
            judge_model=args.judge_model,
            judge_api_base=args.judge_api_base,
            max_samples=args.max_samples,
            pass_threshold=args.pass_threshold,
            output_dir=args.output_dir,
            iteration=args.iteration,
            topic=args.topic,
        )
    except Exception:
        sys.stderr.write("\n[eval] Fatal error:\n")
        traceback.print_exc()
        sys.exit(1)
