"""
autoresearch/cli.py
-------------------
Unified single-command CLI for UltimateDocResearcher.

This is the Phase 13 entry point.  It wires together:
  collect → [analyze] → prepare → eval → suggest → package

Usage examples:

  # Minimal: generate deliverables from an existing corpus
  python -m autoresearch.cli --topic "RAG architecture"

  # Full pipeline (collect + analyze + package)
  python -m autoresearch.cli \\
      --topic "Claude tool use patterns" \\
      --full \\
      --pdf-dir papers/

  # Incremental: only re-collect changed/new files
  python -m autoresearch.cli \\
      --topic "Claude tool use patterns" \\
      --collect --incremental \\
      --pdf-dir papers/

  # Named output directory
  python -m autoresearch.cli \\
      --topic "multi-tenant RAG" \\
      --output results/rag-research \\
      --model claude-3-5-haiku-20241022

Steps in the pipeline
---------------------
  collect   Run document collector (full or incremental)
  analyze   Analyze corpus quality and filter noise
  prepare   Generate Q&A pairs for eval
  package   Generate multi-format deliverables (SUMMARY, ARCH, CODE, …)

All steps are opt-in individually or via --full.  The minimum useful
invocation is `--topic` alone (generates from an existing corpus).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Step runners ──────────────────────────────────────────────────────────────

def _step_collect(args) -> bool:
    """Run collection (incremental or full). Returns True on success."""
    if args.incremental:
        print("\n[cli] Step 1/4 — Incremental collect")
        from autoresearch.incremental_collect import IncrementalCollector
        ic = IncrementalCollector(
            data_dir=args.data_dir,
            pdf_dir=args.pdf_dir,
            verbose=True,
        )
        new, skipped = ic.run(
            topic=args.topic,
            extra_urls=args.urls or [],
            github_repos=args.github or [],
            queries=args.queries or [],
            force=args.force_recollect,
        )
        print(f"[cli]   New: {new} docs, Skipped: {skipped} cached")
    else:
        print("\n[cli] Step 1/4 — Full collect")
        import subprocess
        cmd = [
            sys.executable, "-m", "collector.ultimate_collector",
            "--pdf-dir", str(ROOT / args.pdf_dir),
            "--output-dir", str(ROOT / args.data_dir),
        ]
        if args.queries:
            cmd += ["--queries"] + args.queries
        if args.github:
            cmd += ["--github"] + args.github
        if args.urls:
            cmd += ["--urls"] + args.urls
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print("[cli]   ⚠️  Collector exited with errors — continuing")
    return True


def _step_analyze(args) -> bool:
    """Analyze corpus. Returns True on success."""
    print("\n[cli] Step 2/4 — Analyze corpus")
    corpus_path = ROOT / args.corpus
    if not corpus_path.exists():
        print(f"[cli]   ⚠️  Corpus not found at {corpus_path}, skipping analyze")
        return False
    try:
        from collector.analyzer import analyze_corpus
        report = analyze_corpus(str(corpus_path), output_dir=str(ROOT / args.data_dir), verbose=True)
        total = report.get("total_docs", "?")
        kept = report.get("kept_docs", "?")
        print(f"[cli]   Corpus: {total} docs → {kept} after filtering")
    except Exception as exc:
        print(f"[cli]   ⚠️  Analyze error: {exc}")
        return False
    return True


def _step_prepare(args) -> bool:
    """Generate Q&A pairs."""
    print("\n[cli] Step 3/4 — Prepare Q&A")
    corpus_path = ROOT / args.corpus
    if not corpus_path.exists():
        print(f"[cli]   ⚠️  Corpus not found, skipping prepare")
        return False
    try:
        import subprocess
        cmd = [
            sys.executable, "-m", "autoresearch.prepare",
            "--corpus", str(corpus_path),
            "--output-dir", str(ROOT / args.data_dir),
            "--max-pairs", str(args.max_pairs),
        ]
        if args.model:
            cmd += ["--model", args.model]
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print("[cli]   ⚠️  Prepare exited with errors")
            return False
    except Exception as exc:
        print(f"[cli]   ⚠️  Prepare error: {exc}")
        return False
    return True


def _step_package(args) -> int:
    """Generate multi-format research deliverables. Returns exit code."""
    print("\n[cli] Step 4/4 — Generate research package")
    import subprocess
    cmd = [
        sys.executable, "-m", "autoresearch.research",
        "--topic", args.topic,
        "--corpus", str(ROOT / args.corpus),
    ]
    if args.output:
        cmd += ["--output-dir", args.output]
    if args.model:
        cmd += ["--model", args.model]
    if args.no_code:
        cmd.append("--no-code")
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autoresearch.cli",
        description="UltimateDocResearcher — single-command research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Core
    parser.add_argument("--topic",    required=True, help="Research topic")
    parser.add_argument("--corpus",   default="data/all_docs_cleaned.txt",
                        help="Cleaned corpus path (default: data/all_docs_cleaned.txt)")
    parser.add_argument("--output",   default=None,
                        help="Output directory (default: results/<slug>-<ts>/)")
    parser.add_argument("--model",    default=None,
                        help="LLM model for generation (default: auto-detect)")

    # Pipeline steps
    parser.add_argument("--full",       action="store_true",
                        help="Run full pipeline: collect + analyze + prepare + package")
    parser.add_argument("--collect",    action="store_true", help="Run document collection")
    parser.add_argument("--analyze",    action="store_true", help="Analyze corpus")
    parser.add_argument("--prepare",    action="store_true", help="Generate Q&A pairs")
    parser.add_argument("--package",    action="store_true", help="Generate deliverable package")

    # Collection options
    parser.add_argument("--incremental", action="store_true",
                        help="Incremental collection: skip unchanged sources")
    parser.add_argument("--force-recollect", action="store_true",
                        help="With --incremental: ignore cached hashes and re-collect all")
    parser.add_argument("--pdf-dir",    default="papers/",
                        help="PDF/text directory (default: papers/)")
    parser.add_argument("--data-dir",   default="data/",
                        help="Data directory (default: data/)")
    parser.add_argument("--urls",       nargs="*", default=None, help="Extra URLs to collect")
    parser.add_argument("--github",     nargs="*", default=None, help="GitHub repos (user/repo)")
    parser.add_argument("--queries",    nargs="*", default=None, help="Web search queries")

    # Package options
    parser.add_argument("--no-code",    action="store_true", help="Skip code suggestions")
    parser.add_argument("--max-pairs",  type=int, default=50,
                        help="Max Q&A pairs to generate (default: 50)")

    args = parser.parse_args()

    # --full implies all steps
    if args.full:
        args.collect = args.analyze = args.prepare = args.package = True

    # Default: at minimum, generate the package
    if not any([args.collect, args.analyze, args.prepare, args.package, args.full]):
        args.package = True

    start = time.time()
    print(f"\n{'='*60}")
    print(f"  UltimateDocResearcher — Phase 13 CLI")
    print(f"  Topic: {args.topic}")
    print(f"{'='*60}\n")

    exit_code = 0

    if args.collect:
        _step_collect(args)

    if args.analyze or args.collect:
        _step_analyze(args)

    if args.prepare:
        _step_prepare(args)

    if args.package or not any([args.collect, args.analyze, args.prepare]):
        exit_code = _step_package(args)

    elapsed = time.time() - start
    print(f"\n[cli] Pipeline complete in {elapsed:.1f}s")
    return exit_code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
