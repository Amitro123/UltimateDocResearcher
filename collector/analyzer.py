"""
analyzer.py — Text analysis and quality filtering for collected documents.

Runs after UltimateCollector to:
  1. Score document quality (length, language, information density)
  2. Chunk long docs into training-sized windows
  3. Generate a summary report
  4. Write the final all_docs.txt ready for autoresearch/prepare.py
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class DocStats:
    title: str
    source: str
    chars: int
    words: int
    sentences: int
    quality_score: float  # 0.0 – 1.0


def score_document(text: str) -> float:
    """
    Heuristic quality score (0-1):
      • Penalise very short texts
      • Reward higher type-token ratio (vocabulary diversity)
      • Penalise boilerplate patterns (cookie notices, nav bars, etc.)
    """
    if not text.strip():
        return 0.0

    words = text.split()
    n_words = len(words)
    if n_words < 50:
        return 0.1

    # Type-token ratio (capped at 0.5 → 1.0 range)
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
    return max(0.0, min(1.0, score))


def chunk_text(
    text: str,
    chunk_size: int = 2048,
    overlap: int = 128,
) -> List[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` chars.
    Respects paragraph boundaries where possible.
    """
    paragraphs = re.split(r"\n\n+", text.strip())
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(current.strip())
            # Overlap: keep last `overlap` chars
            current = current[-overlap:] + "\n\n" + para
        else:
            current = current + ("\n\n" if current else "") + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def analyze_corpus(all_docs_path: str | Path, output_dir: str | Path = "data/") -> dict:
    """
    Read all_docs.txt, compute stats, filter low-quality docs, re-chunk,
    and write cleaned all_docs.txt + report.json.
    """
    path = Path(all_docs_path)
    output_dir = Path(output_dir)
    raw_text = path.read_text(encoding="utf-8")

    # Split by doc separator
    raw_docs = raw_text.split("<DOC_SEP>")

    stats_list: List[DocStats] = []
    good_chunks: List[str] = []

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
        score = score_document(body)

        stats_list.append(DocStats(
            title=title,
            source=source,
            chars=len(body),
            words=len(words),
            sentences=sentences,
            quality_score=round(score, 3),
        ))

        if score >= 0.25:
            chunks = chunk_text(body)
            good_chunks.extend(chunks)

    # Write cleaned corpus
    cleaned_path = output_dir / "all_docs_cleaned.txt"
    with cleaned_path.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(good_chunks))

    # Write report
    report = {
        "total_docs": len(stats_list),
        "docs_passing_filter": sum(1 for s in stats_list if s.quality_score >= 0.25),
        "total_chars_raw": sum(s.chars for s in stats_list),
        "total_chars_cleaned": cleaned_path.stat().st_size,
        "total_chunks": len(good_chunks),
        "avg_quality_score": round(statistics.mean(s.quality_score for s in stats_list), 3) if stats_list else 0,
        "source_breakdown": _source_breakdown(stats_list),
        "doc_stats": [
            {"title": s.title, "source": s.source, "words": s.words, "quality": s.quality_score}
            for s in sorted(stats_list, key=lambda x: x.quality_score, reverse=True)[:20]
        ],
    }
    report_path = output_dir / "corpus_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"✅ Analyzer complete:")
    print(f"   {report['total_docs']} docs → {report['docs_passing_filter']} pass quality filter")
    print(f"   {report['total_chunks']} chunks → {cleaned_path}")
    print(f"   Avg quality score: {report['avg_quality_score']}")
    return report


def _source_breakdown(stats: List[DocStats]) -> dict:
    breakdown: dict = {}
    for s in stats:
        breakdown.setdefault(s.source, 0)
        breakdown[s.source] += 1
    return breakdown


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/all_docs.txt")
    parser.add_argument("--output-dir", default="data/")
    args = parser.parse_args()
    analyze_corpus(args.input, args.output_dir)
