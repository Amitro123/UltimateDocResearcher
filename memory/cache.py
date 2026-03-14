"""
memory/cache.py
---------------
Prompt caching: exact match first, fuzzy match (similarity) second.

Avoids redundant LLM calls by storing prompt→response pairs in a
JSONL file and looking up identical or near-identical prompts before
making new API requests.

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

    # Fuzzy lookup (similarity ≥ threshold)
    hit = cache.get_fuzzy("Explain LoRA for fine-tuning LLMs", threshold=0.85)
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from memory.memory import topic_similarity   # reuse pure-Python cosine sim

# ── Default paths ─────────────────────────────────────────────────────────────

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "cache"


class PromptCache:
    """
    JSONL-backed prompt cache with exact and fuzzy lookup.

    Each entry stored as one JSON line:
        {
          "hash":      "<sha1 of prompt>",
          "prompt":    "<original prompt>",
          "response":  "<cached response>",
          "model":     "<model string>",
          "timestamp": "<ISO UTC>",
          "hits":      <int>
        }
    """

    def __init__(self, cache_dir: str | Path = _DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.cache_dir / "prompts.jsonl"
        self._entries: list[dict] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def _save_entry(self, entry: dict) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _rewrite(self) -> None:
        """Rewrite the full JSONL file (used after hit count updates)."""
        with self._path.open("w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ── Public API ────────────────────────────────────────────────────────

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha1(prompt.strip().encode()).hexdigest()[:16]

    def get(self, prompt: str, model: str = "") -> Optional[dict]:
        """
        Exact lookup by prompt hash.
        Optionally filters by model string if provided.
        Returns the cache entry dict or None.
        """
        h = self._hash(prompt)
        for entry in self._entries:
            if entry["hash"] == h:
                if model and entry.get("model", "") != model:
                    continue
                entry["hits"] = entry.get("hits", 0) + 1
                self._rewrite()
                return entry
        return None

    def get_fuzzy(
        self,
        prompt: str,
        threshold: float = 0.85,
        model: str = "",
        max_age_hours: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Fuzzy lookup: return the most similar cached entry above `threshold`.
        Falls back to None if nothing is similar enough.
        """
        best_entry: Optional[dict] = None
        best_sim = 0.0

        cutoff_ts: Optional[float] = None
        if max_age_hours is not None:
            cutoff_ts = time.time() - max_age_hours * 3600

        for entry in self._entries:
            if model and entry.get("model", "") != model:
                continue
            if cutoff_ts is not None:
                # Parse ISO timestamp
                try:
                    ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                    if ts < cutoff_ts:
                        continue
                except (ValueError, KeyError):
                    pass

            sim = topic_similarity(prompt, entry["prompt"])
            if sim > best_sim and sim >= threshold:
                best_sim = sim
                best_entry = entry

        if best_entry:
            best_entry["hits"] = best_entry.get("hits", 0) + 1
            best_entry["_fuzzy_similarity"] = round(best_sim, 3)
            self._rewrite()

        return best_entry

    def set(self, prompt: str, response: str, model: str = "") -> dict:
        """
        Store a prompt→response pair.
        If an identical prompt exists, updates its response in place.
        """
        h = self._hash(prompt)

        # Update existing entry if present
        for entry in self._entries:
            if entry["hash"] == h and entry.get("model", "") == model:
                entry["response"] = response
                entry["timestamp"] = _now()
                self._rewrite()
                return entry

        # New entry
        entry = {
            "hash": h,
            "prompt": prompt,
            "response": response,
            "model": model,
            "timestamp": _now(),
            "hits": 0,
        }
        self._entries.append(entry)
        self._save_entry(entry)
        return entry

    def invalidate(self, prompt: str, model: str = "") -> bool:
        """Remove an entry by exact prompt match. Returns True if removed."""
        h = self._hash(prompt)
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if not (e["hash"] == h and (not model or e.get("model") == model))
        ]
        if len(self._entries) < before:
            self._rewrite()
            return True
        return False

    def clear(self) -> None:
        """Remove all cached entries."""
        self._entries = []
        self._path.write_text("")

    def stats(self) -> dict:
        """Summary stats for the cache."""
        total_hits = sum(e.get("hits", 0) for e in self._entries)
        models = list({e.get("model", "") for e in self._entries if e.get("model")})
        return {
            "total_entries": len(self._entries),
            "total_hits": total_hits,
            "cache_file": str(self._path),
            "models": models,
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"PromptCache(entries={len(self)}, path={self._path})"


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
