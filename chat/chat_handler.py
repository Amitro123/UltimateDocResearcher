"""
chat/chat_handler.py
--------------------
Pipeline orchestrator for the Research Chat interface.

Runs the full research pipeline as subprocess steps, yielding
(event_type, payload) tuples so the Streamlit UI can render
live progress without blocking.

Event types:
  "status"  – human-readable step label (shown in st.status header)
  "log"     – single stdout line from the subprocess
  "done"    – step finished successfully
  "error"   – step failed (message included)
  "result"  – final dict with output file contents
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Generator, List, Optional

ROOT = Path(__file__).resolve().parent.parent


# ── Memory helpers ─────────────────────────────────────────────────────────────

def find_similar_runs(topic: str, threshold: float = 0.75) -> list:
    """Return past runs with topic similarity ≥ threshold."""
    try:
        sys.path.insert(0, str(ROOT))
        from memory.memory import RunMemory
        mem = RunMemory(ROOT / "dashboard" / "runs.db")
        return mem.find_similar(topic, threshold=threshold)
    except Exception:
        return []


# ── Subprocess step runner ─────────────────────────────────────────────────────

def _run_step(
    cmd: List[str],
    label: str,
) -> Generator[tuple, None, None]:
    """
    Run a subprocess and stream its output line by line.
    Raises RuntimeError if the process exits non-zero.
    """
    yield ("status", label)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                yield ("log", line)
        proc.wait()
    except FileNotFoundError as exc:
        yield ("error", f"{label} — command not found: {exc}")
        raise RuntimeError(f"Command not found: {cmd[0]}") from exc

    if proc.returncode != 0:
        yield ("error", f"{label} failed (exit code {proc.returncode})")
        raise RuntimeError(f"Step failed: {label}")

    yield ("done", label)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    topic: str,
    pdf_files=None,           # list of Streamlit UploadedFile objects
    extra_urls: Optional[List[str]] = None,
    github_repos: Optional[List[str]] = None,
    n_suggestions: int = 5,
) -> Generator[tuple, None, None]:
    """
    Full research pipeline. Yields (event_type, payload) tuples.

    Steps:
      1. new_run.py        — workspace reset, archive old papers
      2. Save uploaded PDFs to papers/
      3. ultimate_collector — collect from PDFs + URLs
      4. analyzer          — quality filter, dedup, clean
      5. prepare.py        — generate Q&A pairs
      6. code_suggester    — generate code suggestions
    """
    py = sys.executable

    # ── Step 1: Workspace reset ────────────────────────────────────────────────
    yield from _run_step(
        [py, str(ROOT / "new_run.py"), "--topic", topic, "--keep-cache"],
        "🔄 Resetting workspace",
    )

    # ── Step 2: Save uploaded PDFs to papers/ ─────────────────────────────────
    if pdf_files:
        yield ("status", "📎 Saving uploaded files")
        papers_dir = ROOT / "papers"
        papers_dir.mkdir(exist_ok=True)
        for f in pdf_files:
            data = f.getvalue() if hasattr(f, "getvalue") else f.read()
            dest = papers_dir / f.name
            dest.write_bytes(data)
            yield ("log", f"  Saved {f.name} ({len(data) // 1024} KB)")
        yield ("done", "📎 Files saved")

    # ── Step 3: Collect documents ──────────────────────────────────────────────
    collect_cmd = [
        py, "-m", "collector.ultimate_collector",
        "--pdf-dir", "papers/",
        "--output-dir", "data/",
        "--queries", topic,
    ]
    if extra_urls:
        collect_cmd += ["--urls"] + extra_urls
    if github_repos:
        collect_cmd += ["--github"] + github_repos

    yield from _run_step(collect_cmd, "📚 Collecting documents")

    # ── Step 4: Analyze & clean corpus ────────────────────────────────────────
    yield from _run_step(
        [
            py, "-c",
            "import sys; sys.path.insert(0, '.'); "
            "from collector.analyzer import analyze_corpus; "
            "analyze_corpus('data/all_docs.txt', 'data/')",
        ],
        "🔍 Analyzing corpus",
    )

    corpus_path = ROOT / "data" / "all_docs_cleaned.txt"
    if not corpus_path.exists():
        yield ("error", "Corpus not found after collection — check collector output above")
        return

    corpus_chars = len(corpus_path.read_text(encoding="utf-8", errors="ignore"))
    yield ("log", f"  Corpus: {corpus_chars:,} chars")

    # ── Step 5: Prepare Q&A pairs ─────────────────────────────────────────────
    yield from _run_step(
        [
            py, "-m", "autoresearch.prepare",
            "--corpus", "data/all_docs_cleaned.txt",
            "--max-pairs", "50",
        ],
        "🧠 Preparing Q&A pairs",
    )

    # ── Step 6: Generate code suggestions ─────────────────────────────────────
    yield from _run_step(
        [
            py, "-m", "autoresearch.code_suggester",
            "--corpus", "data/all_docs_cleaned.txt",
            "--topic", topic,
            "--n-suggestions", str(n_suggestions),
        ],
        "💡 Generating code suggestions",
    )

    # ── Collect outputs ────────────────────────────────────────────────────────
    result: dict = {"topic": topic}

    suggestions_path = ROOT / "results" / "code_suggestions.md"
    if suggestions_path.exists():
        result["code_suggestions"] = suggestions_path.read_text(encoding="utf-8")
        result["suggestions_path"] = str(suggestions_path)

    val_path = ROOT / "data" / "val.jsonl"
    if val_path.exists():
        qa_pairs = []
        for line in val_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                qa_pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        result["qa_pairs"] = qa_pairs[:20]

    yield ("result", result)
