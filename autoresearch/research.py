"""
autoresearch/research.py
------------------------
Multi-format research output CLI.

Runs the full research pipeline (collect → analyze → generate deliverables)
and outputs a structured package of Markdown documents tailored to the
research type (code, arch, process, or market survey).

Output per run: results/<slug>-<timestamp>/
  SUMMARY.md          — executive overview
  ARCHITECTURE.md     — system design (arch topics)
  CODE/
    code_suggestions.md — copy-paste code patterns
  IMPLEMENTATION.md   — step-by-step plan
  RISKS.md            — risk register
  BENCHMARKS.md       — comparisons & data (market topics)
  NEXT_STEPS.md       — prioritised actions
  metadata.json       — run metadata

Usage:
    # Minimal — just generate deliverables from an existing corpus
    python -m autoresearch.research --topic "multi-tenant RAG"

    # Full pipeline — collect, analyze, then generate
    python -m autoresearch.research \\
        --topic "Claude tool use patterns" \\
        --collect \\
        --pdf-dir papers/ \\
        --iterations 1

    # Use specific model
    python -m autoresearch.research \\
        --topic "LLM evaluation frameworks" \\
        --model gemini-2.5-flash-lite

    # Custom output directory
    python -m autoresearch.research \\
        --topic "RAG architecture" \\
        --output-dir results/my-rag-research
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run_collect(topic: str, pdf_dir: str, output_dir: str) -> None:
    """Run the document collector pipeline."""
    import subprocess
    cmd = [
        sys.executable, "-m", "collector.ultimate_collector",
        "--pdf-dir", pdf_dir,
        "--output-dir", output_dir,
    ]
    print(f"[research] Running collector: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("[research] ⚠️  Collector exited with errors — continuing with existing corpus")


def _run_analyze(corpus_path: str, output_dir: str) -> dict:
    """Run corpus analysis and return the report dict."""
    from collector.analyzer import analyze_corpus
    print(f"[research] Analyzing corpus: {corpus_path}")
    report = analyze_corpus(corpus_path, output_dir=output_dir, verbose=True)
    return report


def _print_package_summary(pkg) -> None:
    """Print a human-readable summary of the generated package."""
    print("\n" + "═" * 60)
    print(f"  Research Package: {pkg.topic}")
    print(f"  Type:    {pkg.research_type}")
    print(f"  Run ID:  {pkg.run_id}")
    print(f"  Output:  {pkg.output_dir}")
    print("═" * 60)
    print("\nFiles generated:")
    for name, path in sorted(pkg.files.items()):
        size_kb = path.stat().st_size / 1024
        print(f"  ✅  {name:<35} ({size_kb:.1f} KB)")
    if pkg.errors:
        print("\nErrors:")
        for name, err in pkg.errors.items():
            print(f"  ❌  {name}: {err}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate multi-format research deliverables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Core args
    parser.add_argument(
        "--topic", required=True,
        help="Research topic (e.g. 'multi-tenant RAG with Claude tool use')"
    )
    parser.add_argument(
        "--corpus", default="data/all_docs_cleaned.txt",
        help="Path to cleaned corpus file (default: data/all_docs_cleaned.txt)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/<topic-slug>-<timestamp>/)"
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model to use (default: auto-detect best available)"
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Custom run identifier (auto-generated if omitted)"
    )

    # Pipeline control
    parser.add_argument(
        "--collect", action="store_true",
        help="Run the document collector before generating deliverables"
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="With --collect: use incremental mode (skip unchanged sources)"
    )
    parser.add_argument(
        "--force-recollect", action="store_true",
        help="With --collect --incremental: ignore cached hashes and re-collect all"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run corpus analysis before generating deliverables"
    )
    parser.add_argument(
        "--pdf-dir", default="papers/",
        help="PDF directory for collector (default: papers/)"
    )
    parser.add_argument(
        "--data-dir", default="data/",
        help="Data directory for collector output (default: data/)"
    )
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="Research iterations (used with --collect, default: 1)"
    )

    # Input type
    parser.add_argument(
        "--input-type", default=None,
        choices=["error_log", "codebase", "paper", "website", "text"],
        help=(
            "Override input type detection. "
            "'error_log' → ROOT_CAUSE + FIX_STEPS + PREVENTION; "
            "'paper' → SUMMARY + KEY_TAKEAWAYS + BENCHMARKS; "
            "'codebase' → ARCHITECTURE + TESTS + CODE; "
            "'website' → FLOW + INTEGRATION; "
            "'text' → auto topic-based classification (default)"
        ),
    )

    # Output options
    parser.add_argument(
        "--no-code", action="store_true",
        help="Skip code_suggester (don't generate CODE/ deliverable)"
    )
    parser.add_argument(
        "--corpus-chars", type=int, default=40_000,
        help="Max corpus chars to send to LLM (default: 40000)"
    )
    parser.add_argument(
        "--json-summary", action="store_true",
        help="Print a JSON summary of the package to stdout after completion"
    )

    args = parser.parse_args()

    corpus_path = ROOT / args.corpus
    data_dir = ROOT / args.data_dir

    # ── Optional: collect ──────────────────────────────────────────────────
    if args.collect:
        if args.incremental:
            from autoresearch.incremental_collect import IncrementalCollector
            ic = IncrementalCollector(
                data_dir=str(data_dir),
                pdf_dir=str(ROOT / args.pdf_dir),
                verbose=True,
            )
            new, skipped = ic.run(
                topic=args.topic,
                force=getattr(args, "force_recollect", False),
            )
            print(f"[research] Incremental collect: {new} new docs, {skipped} skipped")
        else:
            _run_collect(
                topic=args.topic,
                pdf_dir=str(ROOT / args.pdf_dir),
                output_dir=str(data_dir),
            )

    # ── Optional: analyze ─────────────────────────────────────────────────
    if args.analyze or args.collect:
        if corpus_path.exists():
            _run_analyze(str(corpus_path), str(data_dir))
        else:
            print(f"[research] ⚠️  Corpus not found at {corpus_path} — skipping analysis")

    # ── Pre-flight check ──────────────────────────────────────────────────
    if not corpus_path.exists():
        print(f"[research] ❌ Corpus not found: {corpus_path}")
        print("[research] Run the collector first:")
        print(f"  python -m collector.ultimate_collector --pdf-dir papers/ --output-dir data/")
        return 1

    # ── Generate deliverables ─────────────────────────────────────────────
    print(f"\n[research] Topic: {args.topic}")
    print(f"[research] Corpus: {corpus_path}")

    from research_deliverables.generators import generate_deliverables

    try:
        pkg = generate_deliverables(
            topic=args.topic,
            corpus_path=corpus_path,
            output_dir=args.output_dir,
            model=args.model,
            run_id=args.run_id,
            max_corpus_chars=args.corpus_chars,
            include_code=not args.no_code,
            input_type=getattr(args, "input_type", None),
        )
    except Exception as exc:
        import traceback
        print(f"\n[research] ❌ Failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    _print_package_summary(pkg)

    if args.json_summary:
        print(json.dumps(pkg.metadata, indent=2))

    # Register in memory DB if available
    try:
        from memory.memory import RunMemory
        mem = RunMemory(ROOT / "dashboard" / "runs.db")
        # Store the package output dir as notes, suggestions_path as CODE file
        code_path = pkg.files.get("CODE/code_suggestions.md")
        mem.finish_run(
            run_id=None,   # new entry (no pre-registered run)
            topic=pkg.topic,
            suggestions_path=str(code_path.relative_to(ROOT)) if code_path else None,
            notes=f"Multi-format package: {pkg.output_dir.relative_to(ROOT)}",
        )
    except Exception:
        pass  # memory DB is optional

    return 0 if not pkg.errors else 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
