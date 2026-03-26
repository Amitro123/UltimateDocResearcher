"""
dashboard/seed_demo.py
----------------------
Populate runs.db with realistic demo data for dashboard development/testing.

Run once:
    python dashboard/seed_demo.py

Adds 8 completed runs across 4 topics with per-iteration metrics, so all
dashboard views have data to render without needing real research runs.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.memory import RunMemory

DEMO_RUNS = [
    {
        "topic": "Claude tool use patterns for agentic workflows",
        "days_ago": 14,
        "iterations": 5,
        "avg_score": 0.81,
        "pass_rate": 0.74,
        "weighted_eval": 4.1,
        "n_suggestions": 5,
        "judge_model": "ollama:llama3.2",
        "notes": "Good coverage of tool_use blocks and multi-turn loops.",
    },
    {
        "topic": "LoRA fine-tuning best practices with unsloth",
        "days_ago": 10,
        "iterations": 3,
        "avg_score": 0.77,
        "pass_rate": 0.68,
        "weighted_eval": 3.8,
        "n_suggestions": 5,
        "judge_model": "ollama:llama3.2",
        "notes": "Corpus was thin on QLoRA specifics — re-collect recommended.",
    },
    {
        "topic": "Building SKILL.md files for Claude desktop skills",
        "days_ago": 7,
        "iterations": 4,
        "avg_score": 0.88,
        "pass_rate": 0.82,
        "weighted_eval": 4.4,
        "n_suggestions": 7,
        "judge_model": "claude-3-5-haiku-20241022",
        "notes": "Strong corpus from the official skills guide PDF.",
    },
    {
        "topic": "Building SKILL.md files for Claude Code integration",
        "days_ago": 6,
        "iterations": 2,
        "avg_score": 0.85,
        "pass_rate": 0.79,
        "weighted_eval": 4.2,
        "n_suggestions": 5,
        "judge_model": "ollama:llama3.2",
        "notes": "Very similar to previous skills run — consider merging topics.",
    },
    {
        "topic": "MCP server development and tool registration",
        "days_ago": 4,
        "iterations": 5,
        "avg_score": 0.79,
        "pass_rate": 0.71,
        "weighted_eval": 3.9,
        "n_suggestions": 5,
        "judge_model": "ollama:llama3.2",
        "notes": "",
    },
    {
        "topic": "Prompt caching strategies for Anthropic API cost reduction",
        "days_ago": 3,
        "iterations": 3,
        "avg_score": 0.72,
        "pass_rate": 0.65,
        "weighted_eval": 3.6,
        "n_suggestions": 4,
        "judge_model": "ollama:llama3.2",
        "notes": "Low freshness score — corpus predates cache_control param.",
    },
    {
        "topic": "Claude SKILL.md trigger description optimization",
        "days_ago": 1,
        "iterations": 3,
        "avg_score": 0.91,
        "pass_rate": 0.87,
        "weighted_eval": 4.6,
        "n_suggestions": 6,
        "judge_model": "claude-3-5-haiku-20241022",
        "notes": "Best run so far. New description regex approach works well.",
    },
    {
        "topic": "Agentic loop design with Claude tool use",
        "days_ago": 0,
        "iterations": 0,
        "avg_score": None,
        "pass_rate": None,
        "weighted_eval": None,
        "n_suggestions": None,
        "judge_model": "ollama:llama3.2",
        "status": "running",
        "notes": "In progress.",
    },
]


def seed(db_path: str | Path | None = None) -> None:
    mem = RunMemory(db_path) if db_path else RunMemory()

    rng = random.Random(42)
    base_ts = datetime.now(timezone.utc)

    for run in DEMO_RUNS:
        ts = base_ts - timedelta(days=run["days_ago"])
        status = run.get("status", "completed")

        # Manually insert with controlled timestamp
        cur = mem._conn.execute(
            """INSERT INTO runs
               (topic, timestamp, status, judge_model, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (run["topic"], ts.isoformat(timespec="seconds"), status,
             run["judge_model"], run.get("notes", "")),
        )
        run_id = cur.lastrowid
        mem._conn.commit()

        if status == "completed":
            mem.finish_run(
                run_id,
                status="completed",
                iterations=run["iterations"],
                avg_score=run["avg_score"],
                pass_rate=run["pass_rate"],
                weighted_eval=run["weighted_eval"],
                corpus_chars=rng.randint(30_000, 120_000),
                n_suggestions=run["n_suggestions"],
                results_path="results/results.tsv",
                eval_path="results/eval_report.json",
                suggestions_path="results/code_suggestions.md",
                notes=run.get("notes", ""),
            )

            # Per-iteration metrics — simulate gradual improvement
            base_score = rng.uniform(0.55, 0.70)
            for it in range(1, run["iterations"] + 1):
                it_ts = ts + timedelta(minutes=it * 15)
                score = min(base_score + it * rng.uniform(0.02, 0.06), 0.98)
                mem._conn.execute(
                    """INSERT INTO run_metrics
                       (run_id, iteration, train_loss, val_loss, val_score,
                        judge_pass_rate, judge_avg_score, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id, it,
                        round(rng.uniform(0.8, 1.5) - it * 0.1, 4),
                        round(rng.uniform(0.9, 1.6) - it * 0.08, 4),
                        round(score, 4),
                        round(rng.uniform(0.55, 0.75) + it * 0.03, 3),
                        round(rng.uniform(2.8, 3.5) + it * 0.15, 2),
                        it_ts.isoformat(timespec="seconds"),
                    ),
                )
            mem._conn.commit()

    print(f"✅ Seeded {len(DEMO_RUNS)} demo runs into {mem.db_path}")
    mem.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    seed()
