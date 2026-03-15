"""
memory/cache.py
---------------
Prompt caching: exact match first, fuzzy match (similarity) second.

SQLite-backed for O(1) hit-count updates — replaces the original JSONL
implementation which rewrote the entire file on every cache hit (O(n) I/O).

Usage:
    from memory.cache import PromptCache

    cache = PromptCache()

    # Exact lookup
    hit = cache.get("What is LoRA fine-tuning?")
    if hit:
        response = hit["response"]
    else:
        response = call_llm(...)
        cache.set("What is LoRA fine-tuning?", response, model="ollama:llama3.2")

    # Fuzzy lookup (similarity >= threshold)
    hit = cache.get_fuzzy("Explain LoRA for fine-tuning LLMs", threshold=0.85)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memory.memory import topic_similarity  # reuse pure-Python cosine sim

# -- Default paths -------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "cache"
_DB_NAME = "prompts.db"
_LEGACY_JSONL_NAME = "prompts.jsonl"


class PromptCache:
    """
    SQLite-backed prompt cache with exact and fuzzy lookup.

    Replaces the original JSONL implementation which rewrote the entire file
    on every cache hit (O(n) I/O).  SQLite gives O(1) hit-count updates via a
    single UPDATE statement and avoids loading all prompts into memory for
    exact lookups.

    Schema
    ------
    CREATE TABLE prompts (
        hash      TEXT NOT NULL,
        model     TEXT NOT NULL DEFAULT '',
        prompt    TEXT NOT NULL,
        response  TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        hits      INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (hash, model)
    )

    Migration
    ---------
    On first startup after upgrading, any existing prompts.jsonl is imported
    automatically and renamed to prompts.jsonl.migrated.
    """

    def __init__(self, cache_dir: str | Path = _DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.cache_dir / _DB_NAME
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._migrate_jsonl()

    # -- Schema & migration ----------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    hash      TEXT NOT NULL,
                    model     TEXT NOT NULL DEFAULT '',
                    prompt    TEXT NOT NULL,
                    response  TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    hits      INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (hash, model)
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model ON prompts(model)"
            )

    def _migrate_jsonl(self) -> None:
        """Import entries from the legacy JSONL file, then rename it."""
        legacy = self.cache_dir / _LEGACY_JSONL_NAME
        if not legacy.exists():
            return
        imported = 0
        try:
            with legacy.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        with self._conn:
                            self._conn.execute(
                                "INSERT OR IGNORE INTO prompts "
                                "(hash, model, prompt, response, timestamp, hits) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    e.get("hash", self._hash(e.get("prompt", ""))),
                                    e.get("model", ""),
                                    e.get("prompt", ""),
                                    e.get("response", ""),
                                    e.get("timestamp", _now()),
                                    e.get("hits", 0),
                                ),
                            )
                        imported += 1
                    except (json.JSONDecodeError, sqlite3.Error):
                        pass
            legacy.rename(legacy.with_suffix(".jsonl.migrated"))
            print(
                f"[cache] Migrated {imported} entries from {legacy.name} -> {_DB_NAME}",
                flush=True,
            )
        except Exception as exc:
            print(f"[cache] JSONL migration warning: {exc}")

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha1(prompt.strip().encode()).hexdigest()[:16]

    # -- Public API ------------------------------------------------------------

    def get(self, prompt: str, model: str = "") -> Optional[dict]:
        """
        Exact lookup by prompt hash + model.
        Increments hit count in O(1) via a targeted UPDATE.
        Returns the cache entry dict or None.
        """
        h = self._hash(prompt)
        row = self._conn.execute(
            "SELECT * FROM prompts WHERE hash = ? AND model = ?", (h, model)
        ).fetchone()
        if row is None:
            return None
        with self._conn:
            self._conn.execute(
                "UPDATE prompts SET hits = hits + 1 WHERE hash = ? AND model = ?",
                (h, model),
            )
        result = dict(row)
        result["hits"] = result["hits"] + 1  # reflect the increment
        return result

    def get_fuzzy(
        self,
        prompt: str,
        threshold: float = 0.85,
        model: str = "",
        max_age_hours: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Fuzzy lookup: return the most similar cached entry above threshold.
        Loads prompts from SQLite for cosine comparison -- unavoidable without
        a vector index, but hit-count updates remain O(1).
        """
        query = "SELECT * FROM prompts WHERE model = ?"
        params: list = [model]

        if max_age_hours is not None:
            cutoff = datetime.fromtimestamp(
                time.time() - max_age_hours * 3600, tz=timezone.utc
            ).isoformat(timespec="seconds")
            query += " AND timestamp >= ?"
            params.append(cutoff)

        rows = self._conn.execute(query, params).fetchall()

        best_row: Optional[sqlite3.Row] = None
        best_sim = 0.0
        for row in rows:
            sim = topic_similarity(prompt, row["prompt"])
            if sim > best_sim and sim >= threshold:
                best_sim = sim
                best_row = row

        if best_row is None:
            return None

        with self._conn:
            self._conn.execute(
                "UPDATE prompts SET hits = hits + 1 WHERE hash = ? AND model = ?",
                (best_row["hash"], best_row["model"]),
            )
        result = dict(best_row)
        result["hits"] = result["hits"] + 1
        result["_fuzzy_similarity"] = round(best_sim, 3)
        return result

    def set(self, prompt: str, response: str, model: str = "") -> dict:
        """
        Store a prompt->response pair.
        Updates response + timestamp if the hash+model already exists.
        """
        h = self._hash(prompt)
        ts = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO prompts (hash, model, prompt, response, timestamp, hits)
                VALUES (?, ?, ?, ?, ?, 0)
                ON CONFLICT(hash, model) DO UPDATE SET
                    response  = excluded.response,
                    timestamp = excluded.timestamp
                """,
                (h, model, prompt, response, ts),
            )
        return {
            "hash": h, "model": model, "prompt": prompt,
            "response": response, "timestamp": ts, "hits": 0,
        }

    def invalidate(self, prompt: str, model: str = "") -> bool:
        """Remove an entry by exact prompt match. Returns True if removed."""
        h = self._hash(prompt)
        with self._conn:
            if model:
                cur = self._conn.execute(
                    "DELETE FROM prompts WHERE hash = ? AND model = ?", (h, model)
                )
            else:
                cur = self._conn.execute(
                    "DELETE FROM prompts WHERE hash = ?", (h,)
                )
        return cur.rowcount > 0

    def clear(self) -> None:
        """Remove all cached entries and reclaim disk space."""
        with self._conn:
            self._conn.execute("DELETE FROM prompts")
        self._conn.execute("VACUUM")

    def stats(self) -> dict:
        """Summary stats for the cache."""
        row = self._conn.execute(
            "SELECT COUNT(*) as n, COALESCE(SUM(hits), 0) as total_hits FROM prompts"
        ).fetchone()
        models = [
            r[0] for r in self._conn.execute(
                "SELECT DISTINCT model FROM prompts WHERE model != ''"
            ).fetchall()
        ]
        return {
            "total_entries": row["n"],
            "total_hits": row["total_hits"],
            "cache_file": str(self._db_path),
            "models": models,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM prompts").fetchone()
        return row[0]

    def __repr__(self) -> str:
        return f"PromptCache(entries={len(self)}, path={self._db_path})"

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
