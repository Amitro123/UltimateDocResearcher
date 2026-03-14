"""
memory/memory.py
----------------
SQLite-backed run history and topic similarity search.

Stores every research run (topic, timestamp, scores, paths) in a local
SQLite database and answers "have we researched something similar before?"
using TF-IDF cosine similarity — no ML dependencies required.

Usage:
    from memory.memory import RunMemory

    mem = RunMemory()                        # opens/creates dashboard/runs.db
    run_id = mem.start_run("Claude skills")  # returns int id

    # ... do research ...

    mem.finish_run(run_id, avg_score=0.82, pass_rate=0.78, ...)
    mem.log_iteration(run_id, iteration=1, val_score=0.75, ...)

    # Check for similar past runs before starting a new one
    similar = mem.find_similar("Building skills for Claude", threshold=0.8)
    if similar:
        print(f"Similar run found: {similar[0]['topic']} (score={similar[0]['similarity']:.2f})")
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Default DB path ───────────────────────────────────────────────────────────

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "dashboard" / "runs.db"


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic           TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'running',
    iterations      INTEGER,
    avg_score       REAL,
    pass_rate       REAL,
    weighted_eval   REAL,
    corpus_chars    INTEGER,
    n_suggestions   INTEGER,
    judge_model     TEXT,
    results_path    TEXT,
    eval_path       TEXT,
    suggestions_path TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS run_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    iteration       INTEGER NOT NULL,
    train_loss      REAL,
    val_loss        REAL,
    val_score       REAL,
    judge_pass_rate REAL,
    judge_avg_score REAL,
    timestamp       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_topic     ON runs(topic);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON run_metrics(run_id);
"""


# ── Pure-Python TF-IDF cosine similarity ─────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "on",
    "with", "is", "are", "was", "be", "this", "that", "it", "as",
    "from", "at", "by", "how", "what", "when", "which", "i", "my",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count / total."""
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = max(len(tokens), 1)
    return {t: c / n for t, c in counts.items()}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two TF dicts (no IDF needed for short topics)."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def topic_similarity(a: str, b: str) -> float:
    """Return cosine similarity (0–1) between two topic strings."""
    return _cosine(_tf(_tokenize(a)), _tf(_tokenize(b)))


# ── RunMemory ─────────────────────────────────────────────────────────────────

class RunMemory:
    """
    Persistent run history backed by SQLite.

    Thread-safe for single-process use (SQLite WAL mode).
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────

    def start_run(self, topic: str, judge_model: str = "") -> int:
        """Insert a new run row with status='running'. Returns the run id."""
        cur = self._conn.execute(
            """INSERT INTO runs (topic, timestamp, status, judge_model)
               VALUES (?, ?, 'running', ?)""",
            (topic, _now(), judge_model),
        )
        self._conn.commit()
        return cur.lastrowid

    def finish_run(
        self,
        run_id: int,
        *,
        status: str = "completed",
        iterations: int = 0,
        avg_score: Optional[float] = None,
        pass_rate: Optional[float] = None,
        weighted_eval: Optional[float] = None,
        corpus_chars: Optional[int] = None,
        n_suggestions: Optional[int] = None,
        results_path: str = "",
        eval_path: str = "",
        suggestions_path: str = "",
        notes: str = "",
    ) -> None:
        """Update an existing run row on completion or failure."""
        self._conn.execute(
            """UPDATE runs SET
                status=?, iterations=?, avg_score=?, pass_rate=?,
                weighted_eval=?, corpus_chars=?, n_suggestions=?,
                results_path=?, eval_path=?, suggestions_path=?, notes=?
               WHERE id=?""",
            (
                status, iterations, avg_score, pass_rate,
                weighted_eval, corpus_chars, n_suggestions,
                results_path, eval_path, suggestions_path, notes,
                run_id,
            ),
        )
        self._conn.commit()

    def log_iteration(
        self,
        run_id: int,
        iteration: int,
        *,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_score: Optional[float] = None,
        judge_pass_rate: Optional[float] = None,
        judge_avg_score: Optional[float] = None,
    ) -> None:
        """Append one iteration's metrics row."""
        self._conn.execute(
            """INSERT INTO run_metrics
               (run_id, iteration, train_loss, val_loss, val_score,
                judge_pass_rate, judge_avg_score, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, iteration, train_loss, val_loss, val_score,
             judge_pass_rate, judge_avg_score, _now()),
        )
        self._conn.commit()

    # ── Read ───────────────────────────────────────────────────────────────

    def find_similar(
        self,
        topic: str,
        threshold: float = 0.8,
        max_age_days: Optional[int] = None,
        status: str = "completed",
    ) -> list[dict]:
        """
        Return completed runs whose topic is similar to `topic`.

        Args:
            topic:        Query topic string.
            threshold:    Minimum cosine similarity to include (0–1).
            max_age_days: Only consider runs within this many days.
            status:       Filter by run status (default: 'completed').

        Returns:
            List of dicts sorted by similarity desc, each containing
            all run columns plus a 'similarity' float.
        """
        query = "SELECT * FROM runs WHERE status = ?"
        params: list = [status]

        if max_age_days is not None:
            cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
            query += " AND timestamp >= ?"
            params.append(cutoff)

        rows = self._conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            sim = topic_similarity(topic, row["topic"])
            if sim >= threshold:
                results.append({**dict(row), "similarity": round(sim, 3)})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def recent_runs(self, limit: int = 20) -> list[dict]:
        """Return the N most recent runs, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> Optional[dict]:
        """Fetch a single run by id."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_metrics(self, run_id: int) -> list[dict]:
        """Fetch all iteration metrics for a run, ordered by iteration."""
        rows = self._conn.execute(
            "SELECT * FROM run_metrics WHERE run_id = ? ORDER BY iteration",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_topics(self) -> list[str]:
        """Return distinct topics from all completed runs."""
        rows = self._conn.execute(
            "SELECT DISTINCT topic FROM runs WHERE status='completed'"
        ).fetchall()
        return [r["topic"] for r in rows]

    def stats(self) -> dict:
        """High-level summary stats for the dashboard header."""
        row = self._conn.execute(
            """SELECT
                COUNT(*)                              AS total_runs,
                SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status='failed'    THEN 1 ELSE 0 END) AS failed,
                ROUND(AVG(CASE WHEN avg_score IS NOT NULL THEN avg_score END), 3)
                                                      AS avg_score,
                ROUND(AVG(CASE WHEN pass_rate IS NOT NULL THEN pass_rate END), 3)
                                                      AS avg_pass_rate
               FROM runs"""
        ).fetchone()
        return dict(row) if row else {}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
