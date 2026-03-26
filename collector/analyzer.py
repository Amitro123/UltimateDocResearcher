"""
analyzer.py — Text analysis and quality filtering for collected documents.

Runs after UltimateCollector to:
  1. Score document quality (length, language, information density)
  2. Filter personal documents (invoices, receipts, contracts, CVs)
  3. Filter non-research languages (Hebrew, Arabic, etc.) unless topic-relevant
  4. Chunk long docs into training-sized windows
  5. Generate a summary report
  6. Write the final all_docs.txt ready for autoresearch/prepare.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure emoji in print() doesn't crash narrow-encoding consoles (e.g. Windows cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


@dataclass
class DocStats:
    title: str
    source: str
    chars: int
    words: int
    sentences: int
    quality_score: float        # 0.0 – 1.0
    filter_reason: str = ""     # non-empty when rejected; explains why
    source_type: str = "external"  # "external" | "internal"


# ── Config loader ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load config.yaml from project root. Returns {} if not found."""
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        if cfg_path.exists():
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}


# ── Internal source detection ──────────────────────────────────────────────────

_DEFAULT_INTERNAL_PATTERNS = [
    "ultimatedocresearcher", "ultimate-doc-researcher", "ultimate_doc_researcher",
    "code_review", "agents.md", "claude.md", "e2e_guide",
    "bug_analysis", "walkthrough", "udr",
]

_ALWAYS_EXTERNAL_SOURCES = {"pdf", "web", "github", "reddit", "drive"}


def _build_internal_patterns(cfg: dict) -> list[str]:
    return (
        cfg.get("sources", {}).get("internal_title_patterns")
        or _DEFAULT_INTERNAL_PATTERNS
    )


def is_internal_doc(title: str, source: str, cfg: dict | None = None) -> bool:
    """
    Return True if this document is from the project's own codebase/docs,
    not from external research sources.

    External collector sources (pdf, web, github, reddit, drive) are always
    classified as external regardless of title. Only "local" source files are
    checked against the internal title patterns.
    """
    if cfg is None:
        cfg = {}

    always_ext = set(
        cfg.get("sources", {}).get("always_external_sources")
        or _ALWAYS_EXTERNAL_SOURCES
    )
    if source.lower() in always_ext:
        return False

    # For "local" and unknown sources, check title patterns
    patterns = _build_internal_patterns(cfg)
    title_lower = title.lower()
    return any(p in title_lower for p in patterns)


# ── Personal document detection ───────────────────────────────────────────────

# Patterns that strongly indicate personal/financial documents
_PERSONAL_DOC_PATTERNS = [
    # Invoices & receipts
    r"\binvoice\s*(number|no\.?|#)?\s*[\d\-]+",
    r"\border\s*(number|no\.?|#)\s*[\d\-]+",
    r"\breceipt\s*(number|no\.?|#)?",
    r"\bcheckout\s*order",
    r"\btotal\s+(to\s+pay|amount|due)\b",
    r"\bsubtotal\b.{0,40}\$",
    r"\bpayment\s+(received|due|overdue)\b",  # "payment method" alone is too broad (hits API/tech docs)
    r"\bpayment\s+method\s*:\s*(credit|debit|visa|mastercard|paypal|bank)",  # specific checkout context only
    r"\bship(ping|ped)\s+to\b",
    r"\bvat\s+number\b",
    # Bank / financial statements
    r"\baccount\s+(number|balance|statement)\b",
    r"\btransaction\s+(id|history|date)\b",
    r"\bbank\s+(transfer|statement|account)\b",
    r"\biban\b",
    # Contracts / legal
    r"\bhereby\s+(agree|confirm|acknowledge)\b",
    r"\bterms?\s+and\s+conditions?\b.{0,60}\bsigned?\b",
    r"\bsignature\s+of\s+(employee|employer|party)\b",
    r"\beffective\s+date\s*:\s*\d",
    # CVs / resumes
    r"\b(curriculum\s+vitae|résumé|resume)\b",
    r"\bwork\s+experience\b.{0,200}\beducation\b",
    r"\bskills?\s*:\s*\n.{0,50}\bexperience\b",
    # Medical records
    r"\bpatient\s+(name|id|dob)\b",
    r"\bdiagnosis\b.{0,200}\bprescription\b",
    r"\bhospital\s+(number|admission|discharge)\b",
]

_PERSONAL_DOC_REGEX = re.compile(
    "|".join(_PERSONAL_DOC_PATTERNS), re.I | re.DOTALL
)

# Title-level signals that flag a doc as personal before reading the body
_PERSONAL_TITLE_PATTERNS = re.compile(
    # Underscores and hyphens are common filename separators, so don't use \b
    # after "invoice" etc. — match as long as the keyword appears in the title.
    r"(invoice|receipt|order[\s_\-]\d|\bstatement\b|\bcontract\b|"
    r"offer[\s_\-]letter|curriculum[\s_\-]vitae|cv[\s_\-]\d|\bresume\b|"
    r"\bsalary\b|\bpayslip\b|tax[\s_\-]return|bank[\s_\-]statement|"
    r"medical[\s_\-]record)",
    re.I,
)


def is_personal_document(text: str, title: str = "") -> tuple[bool, str]:
    """
    Return (True, reason) if the document looks like a personal/financial file.
    Checks title first (fast), then body patterns.
    """
    if _PERSONAL_TITLE_PATTERNS.search(title):
        return True, f"personal document title: '{title}'"

    m = _PERSONAL_DOC_REGEX.search(text)
    if m:
        snippet = m.group(0)[:60].replace("\n", " ")
        return True, f"personal document pattern: '{snippet}'"

    return False, ""


# ── Language / script detection ───────────────────────────────────────────────

# Unicode ranges for non-Latin scripts
_SCRIPT_RANGES = {
    "hebrew":   ("\u05d0", "\u05ea"),
    "arabic":   ("\u0600", "\u06ff"),
    "chinese":  ("\u4e00", "\u9fff"),
    "japanese": ("\u3040", "\u30ff"),
    "korean":   ("\uac00", "\ud7af"),
    "cyrillic": ("\u0400", "\u04ff"),
}


def dominant_script(text: str) -> tuple[str, float]:
    """
    Return (script_name, fraction) for the most common non-ASCII script.
    Returns ("latin", 0.0) when the text is predominantly Latin.
    """
    total = max(len(text), 1)
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")
    latin_frac = latin / total

    best_name, best_frac = "latin", 0.0
    for name, (lo, hi) in _SCRIPT_RANGES.items():
        count = sum(1 for c in text if lo <= c <= hi)
        frac = count / total
        if frac > best_frac:
            best_name, best_frac = name, frac

    # If non-Latin script is more than 20% of the text, it dominates
    if best_frac > 0.20 and best_frac > latin_frac * 0.5:
        return best_name, best_frac

    return "latin", latin_frac


def is_non_research_language(text: str, title: str = "") -> tuple[bool, str]:
    """
    Return (True, reason) if the document is predominantly in a non-Latin
    script unlikely to be AI/tech research content.
    """
    script, frac = dominant_script(text)
    if script != "latin" and frac > 0.20:
        pct = int(frac * 100)
        return True, f"{pct}% {script} script — likely not research content"
    return False, ""


# ── Quality scoring ───────────────────────────────────────────────────────────

def score_document(text: str, title: str = "") -> tuple[float, str]:
    """
    Heuristic quality score (0–1) plus a rejection reason string.

    Returns (score, reason) where reason is non-empty when the document
    should be filtered out regardless of the numeric score:
      • Personal/financial documents  → score 0.0
      • Non-research language         → score 0.0
      • Web boilerplate               → penalised
      • Short texts                   → low score
      • Good vocabulary diversity     → high score
    """
    if not text.strip():
        return 0.0, "empty document"

    # Hard filters — run before any scoring
    personal, reason = is_personal_document(text, title)
    if personal:
        return 0.0, reason

    non_latin, reason = is_non_research_language(text, title)
    if non_latin:
        return 0.0, reason

    words = text.split()
    n_words = len(words)
    if n_words < 50:
        return 0.1, "too short"

    # Type-token ratio (vocabulary diversity)
    vocab = len(set(w.lower() for w in words))
    ttr = min(vocab / n_words, 1.0)

    # Boilerplate penalty
    boilerplate_patterns = [
        r"cookie", r"privacy policy", r"terms of service",
        r"subscribe to our newsletter", r"javascript is disabled",
        r"click here to enable", r"404 not found",
    ]
    bp_hits = sum(1 for p in boilerplate_patterns if re.search(p, text, re.I))
    bp_penalty = min(bp_hits * 0.15, 0.5)

    # Length bonus (saturates at ~2000 words)
    length_bonus = min(n_words / 2000, 1.0) * 0.2

    score = ttr * 0.6 + length_bonus - bp_penalty
    return max(0.0, min(1.0, score)), ""


def chunk_text(
    text: str,
    chunk_size: int = 2048,
    overlap: int = 128,
) -> List[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` chars.
    Respects paragraph boundaries where possible.

    Paragraphs larger than `chunk_size` are further split at word boundaries
    so that uniformly-formatted text (no double-newlines) is still chunked
    correctly instead of being returned as a single oversized chunk.
    """
    raw_paragraphs = re.split(r"\n\n+", text.strip())

    # Pre-expand any paragraph that is wider than chunk_size.
    paragraphs: List[str] = []
    for para in raw_paragraphs:
        if len(para) <= chunk_size:
            paragraphs.append(para)
            continue
        # Word-boundary split for oversized paragraph
        start = 0
        while start < len(para):
            end = min(start + chunk_size, len(para))
            if end < len(para):
                # Prefer breaking at last space before the limit
                space = para.rfind(" ", start, end)
                if space > start:
                    end = space
            paragraphs.append(para[start:end].strip())
            start = end

    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(current.strip())
            # Overlap: keep last `overlap` chars.
            # Note: `s[-0:]` in Python returns the full string, not an empty
            # string, so we guard explicitly against overlap == 0.
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail + ("\n\n" if tail else "") + para
        else:
            current = current + ("\n\n" if current else "") + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def analyze_corpus(
    all_docs_path: str | Path,
    output_dir: str | Path = "data/",
    quality_threshold: float = 0.25,
    verbose: bool = True,
) -> dict:
    """
    Read all_docs.txt, compute stats, filter low-quality / personal / non-research
    docs, re-chunk, and write cleaned corpus + corpus_report.json.

    Also writes data/external_docs.txt (external sources only) so that
    code_suggester can prioritize external research over internal project files.

    Filters applied (in order):
      1. Personal/financial documents  (invoices, receipts, contracts, CVs)
      2. Non-research language         (Hebrew, Arabic, etc. > 20% of text)
      3. Quality threshold             (type-token ratio, length, boilerplate)

    Source tagging:
      Each passing document is classified as "external" (PDFs, web, GitHub,
      Reddit) or "internal" (project's own files). Internal docs are included
      in all_docs_cleaned.txt but kept separate in external_docs.txt so the
      LLM can be fed primarily external research.
    """
    cfg = _load_config()
    min_ext_frac = cfg.get("corpus", {}).get("min_external_fraction", 0.30)

    path = Path(all_docs_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_text = path.read_text(encoding="utf-8")

    raw_docs = raw_text.split("<DOC_SEP>")

    stats_list: List[DocStats] = []
    external_chunks: List[str] = []
    internal_chunks: List[str] = []
    filtered_docs: List[dict] = []

    for raw_doc in raw_docs:
        raw_doc = raw_doc.strip()
        if not raw_doc:
            continue

        # Extract header
        m = re.match(r"=== (.+?) \[(\w+)\] ===\n", raw_doc)
        title = m.group(1) if m else "unknown"
        source = m.group(2) if m else "unknown"
        body = raw_doc[m.end():].strip() if m else raw_doc

        words = body.split()
        sentences = len(re.findall(r"[.!?]+", body))
        score, reason = score_document(body, title)
        internal = is_internal_doc(title, source, cfg)

        ds = DocStats(
            title=title,
            source=source,
            chars=len(body),
            words=len(words),
            sentences=sentences,
            quality_score=round(score, 3),
            filter_reason=reason,
            source_type="internal" if internal else "external",
        )
        stats_list.append(ds)

        if score >= quality_threshold and not reason:
            chunks = chunk_text(body)
            if internal:
                internal_chunks.extend(chunks)
            else:
                external_chunks.extend(chunks)
        else:
            effective_reason = reason or f"quality score {score:.2f} < {quality_threshold}"
            filtered_docs.append({"title": title, "reason": effective_reason})
            if verbose:
                print(f"  [filtered] {title[:60]} — {effective_reason}")

    good_chunks = external_chunks + internal_chunks

    # Warn if nothing survived
    if not good_chunks:
        print(
            "\n⚠️  WARNING: 0 documents survived filtering.\n"
            "   Check that --pdf-dir points at a research-specific folder,\n"
            "   not a general Downloads or Documents directory.\n"
            "   All personal/non-research documents were removed.\n"
        )

    # Warn if corpus is mostly internal (corpus contamination)
    total_chunks = len(good_chunks)
    ext_frac = len(external_chunks) / total_chunks if total_chunks else 0.0
    if total_chunks > 0 and ext_frac < min_ext_frac:
        print(
            f"\n⚠️  CORPUS QUALITY WARNING: Only {ext_frac:.0%} of chunks are from external "
            f"sources (minimum recommended: {min_ext_frac:.0%}).\n"
            f"   The corpus is mostly internal project files — code suggestions will\n"
            f"   reflect your existing codebase rather than external research.\n"
            f"   Add external PDFs via --pdf-dir papers/ or URLs via --urls to fix this.\n"
        )

    # Write all cleaned chunks (external first — better for beginning/middle sampling)
    cleaned_path = output_dir / "all_docs_cleaned.txt"
    cleaned_path.write_text("\n\n".join(good_chunks), encoding="utf-8")

    # Write external-only corpus for code_suggester external-first sampling
    external_path = output_dir / "external_docs.txt"
    if external_chunks:
        external_path.write_text("\n\n".join(external_chunks), encoding="utf-8")
    elif external_path.exists():
        external_path.unlink()

    passing = [s for s in stats_list if s.quality_score >= quality_threshold and not s.filter_reason]

    report = {
        "total_docs": len(stats_list),
        "docs_passing_filter": len(passing),
        "docs_filtered_personal": sum(
            1 for s in stats_list
            if "personal document" in s.filter_reason or "script" in s.filter_reason
        ),
        "docs_filtered_quality": sum(
            1 for s in stats_list
            if s.filter_reason == "" and s.quality_score < quality_threshold
        ),
        "total_chars_raw": sum(s.chars for s in stats_list),
        "total_chars_cleaned": cleaned_path.stat().st_size,
        "total_chunks": total_chunks,
        "external_chunks": len(external_chunks),
        "internal_chunks": len(internal_chunks),
        "external_fraction": round(ext_frac, 3),
        "avg_quality_score": round(
            statistics.mean(s.quality_score for s in stats_list), 3
        ) if stats_list else 0,
        "source_breakdown": _source_breakdown(stats_list),
        "source_type_breakdown": {
            "external": sum(1 for s in stats_list if s.source_type == "external" and not s.filter_reason),
            "internal": sum(1 for s in stats_list if s.source_type == "internal" and not s.filter_reason),
        },
        "doc_stats": [
            {
                "title": s.title,
                "source": s.source,
                "source_type": s.source_type,
                "words": s.words,
                "quality": s.quality_score,
                **({"filtered_reason": s.filter_reason} if s.filter_reason else {}),
            }
            for s in sorted(stats_list, key=lambda x: x.quality_score, reverse=True)[:20]
        ],
        "filtered_docs": filtered_docs,
    }
    report_path = output_dir / "corpus_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        personal_n = report["docs_filtered_personal"]
        quality_n = report["docs_filtered_quality"]
        print(f"\n✅ Analyzer complete:")
        print(f"   {report['total_docs']} docs scanned")
        print(f"   {report['docs_passing_filter']} passed  |  "
              f"{personal_n} personal/language filtered  |  "
              f"{quality_n} low-quality filtered")
        print(f"   {total_chunks} chunks → {cleaned_path}")
        print(f"   External: {len(external_chunks)} chunks ({ext_frac:.0%})  |  "
              f"Internal: {len(internal_chunks)} chunks")
        print(f"   Avg quality score: {report['avg_quality_score']}")
        if external_chunks:
            print(f"   External corpus → {external_path}")
        if personal_n:
            print(f"\n   ℹ️  {personal_n} personal documents were removed (invoices, contracts, CVs, etc.).")
            print(f"      Use a dedicated papers/ folder instead of Downloads to avoid this.")

    return report


def _source_breakdown(stats: List[DocStats]) -> dict:
    breakdown: dict = {}
    for s in stats:
        breakdown.setdefault(s.source, 0)
        breakdown[s.source] += 1
    return breakdown


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    import argparse
    parser = argparse.ArgumentParser(description="Analyze and filter research corpus")
    parser.add_argument("--input", default="data/all_docs.txt")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Minimum quality score to keep a document (default: 0.25)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    analyze_corpus(args.input, args.output_dir, args.threshold, verbose=not args.quiet)
