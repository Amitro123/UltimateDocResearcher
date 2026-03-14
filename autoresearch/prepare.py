"""
autoresearch/prepare.py
-----------------------
Wraps (and extends) karpathy/autoresearch's data preparation step.

Takes data/all_docs_cleaned.txt and transforms it into:
  • data/train.jsonl  — (question, answer) pairs for supervised fine-tuning
  • data/val.jsonl    — held-out evaluation set
  • data/program.md   — research program injected into the model

Three Q&A generation backends, in priority order:

  1. NotebookLM  (--source-type notebooklm)
     Upload PDF/URL sources to Google NotebookLM, generate quizzes/flashcards.
     Produces the highest-quality Q&A — NotebookLM deeply understands document
     structure. Requires: `pip install notebooklm-py[browser]` + `notebooklm login`.
     UNOFFICIAL API — may break if Google changes internals. Gracefully falls back.

  2. LLM         (--use-llm --model ollama:llama3.2 | gpt-4o-mini | claude-...)
     Ask any LLM to generate a Q&A pair per corpus chunk.
     Free with Ollama, cheap with cloud APIs.

  3. Heuristic   (default, no dependencies)
     Regex-based sentence extraction — always works, lower quality.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ── Config ─────────────────────────────────────────────────────────────────────

CHUNK_SEP = "\n\n"
DEFAULT_VAL_FRAC = 0.1
DEFAULT_MAX_PAIRS = 500
SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Given a document passage, generate one insightful question and a concise answer."
)


# ── Q&A generation ─────────────────────────────────────────────────────────────

def generate_qa_pair(
    passage: str,
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """
    Generate a Q&A pair from a passage via the unified LLM client.

    Accepts any model string:
      "gpt-4o-mini"               → OpenAI
      "claude-3-5-haiku-20241022" → Anthropic
      "ollama:llama3.2"           → local Ollama (free, no key needed)
    """
    from autoresearch.llm_client import chat, best_available_model
    
    # Auto-detect best model if none provided
    effective_model = model or best_available_model()

    try:
        text = chat(
            messages=[{"role": "user", "content": f"Passage:\n{passage[:3000]}"}],
            model=effective_model,
            system=SYSTEM_PROMPT,
            max_tokens=512,
            temperature=0.7,
            api_base=api_base,
        )
        return _parse_qa(text)
    except Exception as exc:
        print(f"[prepare] QA generation failed: {exc}", file=sys.stderr)
        return None


def _parse_qa(text: str) -> Optional[Tuple[str, str]]:
    """Parse 'Q: ...\nA: ...' format from LLM output."""
    q_match = re.search(r"(?:Q:|Question:)\s*(.+?)(?:\n|$)", text, re.I)
    a_match = re.search(r"(?:A:|Answer:)\s*(.+)", text, re.I | re.DOTALL)
    if q_match and a_match:
        return q_match.group(1).strip(), a_match.group(1).strip()[:1024]
    # Fallback: try splitting on first newline
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:])[:1024]
    return None


# ── NotebookLM Q&A backend ────────────────────────────────────────────────────

def notebooklm_qa_from_sources(
    sources: list[str],
    max_pairs: int = 100,
    quiz_difficulty: str = "medium",
) -> list[tuple[str, str]]:
    """
    Generate Q&A pairs using Google NotebookLM via notebooklm-py.

    Each source can be a local PDF path, a URL, or a Google Drive URL.
    NotebookLM ingests them, then generates a quiz whose questions become
    our training pairs — much higher quality than heuristic extraction.

    Requirements:
        pip install "notebooklm-py[browser]"
        notebooklm login          # one-time browser auth

    Args:
        sources:         list of PDF paths or URLs to ingest
        max_pairs:       cap on Q&A pairs to extract from the quiz
        quiz_difficulty: "easy" | "medium" | "hard"

    Returns:
        list of (question, answer) tuples, or [] on any failure
    """
    try:
        from notebooklm import NotebookLM        # notebooklm-py
    except ImportError:
        print(
            "[prepare] notebooklm-py not installed. "
            "Run: pip install 'notebooklm-py[browser]' && notebooklm login",
            file=sys.stderr,
        )
        return []

    pairs: list[tuple[str, str]] = []
    notebook_id = None

    try:
        client = NotebookLM()

        # Create a fresh notebook for this run
        notebook = client.notebooks.create(f"autoresearch-{os.getpid()}")
        notebook_id = notebook.id
        print(f"[prepare/notebooklm] Created notebook {notebook_id}")

        # Add all sources
        for src in sources:
            try:
                if Path(src).exists():
                    client.sources.add(notebook_id, file=src)
                    print(f"[prepare/notebooklm]   + PDF: {Path(src).name}")
                else:
                    client.sources.add(notebook_id, url=src)
                    print(f"[prepare/notebooklm]   + URL: {src}")
            except Exception as e:
                print(f"[prepare/notebooklm]   ⚠ source skipped ({src}): {e}", file=sys.stderr)

        # Generate quiz and wait for completion
        print(f"[prepare/notebooklm] Generating {quiz_difficulty} quiz…")
        quiz = client.generate(notebook_id, "quiz", difficulty=quiz_difficulty, wait=True)

        # Extract Q&A from quiz items
        items = quiz.get("items") or quiz.get("questions") or []
        for item in items[:max_pairs]:
            q = item.get("question", "").strip()
            # Prefer detailed answer; fall back to first option
            a = (
                item.get("answer")
                or item.get("explanation")
                or item.get("correct_answer")
                or next(iter(item.get("options", {}).values()), "")
            )
            if q and a:
                pairs.append((q, str(a).strip()))

        print(f"[prepare/notebooklm] ✅ {len(pairs)} Q&A pairs from quiz")

    except Exception as exc:
        print(f"[prepare/notebooklm] Failed: {exc}", file=sys.stderr)

    finally:
        # Clean up notebook to avoid cluttering the user's NotebookLM
        if notebook_id:
            try:
                client.notebooks.delete(notebook_id)
                print(f"[prepare/notebooklm] Deleted notebook {notebook_id}")
            except Exception:
                pass

    return pairs


def notebooklm_qa_from_corpus(
    corpus_path: str | Path,
    pdf_sources: list[str] | None = None,
    max_pairs: int = 100,
) -> list[tuple[str, str]]:
    """
    Convenience wrapper: tries NotebookLM, returns [] on any failure so the
    caller can fall through to LLM or heuristic.

    If pdf_sources is given, uses those directly.
    Otherwise looks for PDFs in the same directory as corpus_path.
    """
    sources = pdf_sources or []
    if not sources:
        # Auto-discover PDFs adjacent to the corpus
        corpus_dir = Path(corpus_path).parent
        sources = [str(p) for p in corpus_dir.glob("**/*.pdf")][:10]

    if not sources:
        print(
            "[prepare/notebooklm] No PDF sources found — "
            "pass --pdf-sources or place PDFs in the data directory",
            file=sys.stderr,
        )
        return []

    return notebooklm_qa_from_sources(sources, max_pairs=max_pairs)


# ── Heuristic QA (no API required) ────────────────────────────────────────────

def heuristic_qa(passage: str) -> List[Tuple[str, str]]:
    """
    Generate Q&A pairs without an LLM using heuristics:
      - Extract topic sentences as questions
      - Use surrounding context as answers
    """
    sentences = re.split(r"(?<=[.!?])\s+", passage.strip())
    sentences = [s for s in sentences if len(s) > 40]
    pairs: List[Tuple[str, str]] = []
    for i, sent in enumerate(sentences[:5]):
        # Make a "what/why/how" question from a topic sentence
        q = _sentence_to_question(sent)
        if q:
            # Answer = this + next sentence
            context = " ".join(sentences[i : i + 3])
            pairs.append((q, context))
    return pairs


def _sentence_to_question(sentence: str) -> Optional[str]:
    sentence = sentence.strip().rstrip(".")
    # Passive → "What is..."
    if re.match(r"[A-Z][a-z]+ (is|are|was|were) ", sentence):
        return f"What {sentence[sentence.index(' ')+1:]}?"
    # Simple transformation
    first_word = sentence.split()[0].lower() if sentence.split() else ""
    if first_word in ("the", "a", "an", "this", "these", "it"):
        return f"What is {sentence.lower()}?"
    if first_word in ("we", "researchers", "scientists"):
        return f"How did {sentence.lower()[len(first_word):].strip()}?"
    return f"What do we know about: {sentence}?"


# ── Main prepare pipeline ──────────────────────────────────────────────────────

def prepare(
    corpus_path: str | Path = "data/all_docs_cleaned.txt",
    output_dir: str | Path = "data/",
    program_md: str | Path = "templates/program.md",
    val_frac: float = DEFAULT_VAL_FRAC,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    seed: int = 42,
    # NotebookLM options
    source_type: str = "heuristic",   # "notebooklm" | "llm" | "heuristic"
    pdf_sources: list[str] | None = None,
):
    """
    Full prepare pipeline.

    Args:
        corpus_path:  cleaned corpus from UltimateCollector/analyzer
        output_dir:   where to write train/val jsonl
        program_md:   research program template (injected into prompts)
        val_frac:     fraction held out for validation
        max_pairs:    maximum total Q&A pairs to generate
        use_llm:      shorthand for source_type="llm"
        llm_model:    model for LLM Q&A generation (any llm_client model string)
        seed:         random seed
        source_type:  Q&A backend — "notebooklm" | "llm" | "heuristic"
                      Priority: notebooklm → llm → heuristic (auto-fallback)
        pdf_sources:  explicit PDF paths for NotebookLM source upload
    """
    random.seed(seed)
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    # Resolve source_type shorthand
    if use_llm and source_type == "heuristic":
        source_type = "llm"

    # Load program context
    program_text = ""
    if Path(program_md).exists():
        program_text = Path(program_md).read_text(encoding="utf-8")

    # ── NotebookLM path ────────────────────────────────────────────────────────
    if source_type == "notebooklm":
        print("[prepare] Using NotebookLM backend…")
        nlm_pairs = notebooklm_qa_from_corpus(
            corpus_path, pdf_sources=pdf_sources, max_pairs=max_pairs
        )
        if nlm_pairs:
            raw = corpus_path.read_text(encoding="utf-8")
            chunks = [c.strip() for c in re.split(r"\n\n+", raw) if len(c.strip()) > 150]
            # Attach corpus context to each pair (best-effort chunk match)
            all_pairs = []
            for q, a in nlm_pairs:
                # Find the chunk most likely to be the source
                context = next(
                    (c for c in chunks if any(word in c for word in q.split()[:5])),
                    chunks[0] if chunks else "",
                )
                all_pairs.append(_make_record(q, a, context, program_text))
            print(f"[prepare] NotebookLM produced {len(all_pairs)} pairs ✅")
            return _split_and_write(all_pairs, output_dir, val_frac)
        else:
            print("[prepare] NotebookLM returned 0 pairs — falling back to LLM/heuristic", file=sys.stderr)
            source_type = "llm" if llm_model else "heuristic"

    # Load and split corpus into chunks
    raw = corpus_path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in re.split(r"\n\n+", raw) if len(c.strip()) > 150]
    print(f"[prepare] {len(chunks)} chunks loaded from {corpus_path}")

    # Generate Q&A pairs
    all_pairs: List[dict] = []
    random.shuffle(chunks)

    try:
        for i, chunk in enumerate(chunks):
            if len(all_pairs) >= max_pairs:
                break
            if i % 10 == 0:
                print(f"  Processing chunk {i}/{len(chunks)} (pairs: {len(all_pairs)}/{max_pairs})…", flush=True)

            if source_type == "llm":
                # Ensure we have a model string for the LLM backend
                from autoresearch.llm_client import best_available_model
                effective_llm = llm_model or best_available_model()
                result = generate_qa_pair(chunk, model=effective_llm)
                if result:
                    q, a = result
                    all_pairs.append(_make_record(q, a, chunk, program_text))
            else:
                for q, a in heuristic_qa(chunk):
                    if len(all_pairs) >= max_pairs:
                        break
                    all_pairs.append(_make_record(q, a, chunk, program_text))
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n[prepare] Interrupted by user. Saving partial results...")
        else:
            print(f"\n[prepare] Error during generation: {e}. Saving partial results...", file=sys.stderr)

    if not all_pairs:
        print("[prepare] ⚠ No Q&A pairs were generated.", file=sys.stderr)
        return None, None

    print(f"[prepare] Generated {len(all_pairs)} Q&A pairs")
    return _split_and_write(all_pairs, output_dir, val_frac)


def _split_and_write(
    all_pairs: List[dict], output_dir: Path, val_frac: float
) -> tuple[Path, Path]:
    random.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * val_frac))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    _write_jsonl(train_pairs, train_path)
    _write_jsonl(val_pairs, val_path)

    print(f"[prepare] ✅ {len(train_pairs)} train → {train_path}")
    print(f"[prepare] ✅ {len(val_pairs)} val   → {val_path}")
    return train_path, val_path


def _make_record(
    question: str, answer: str, context: str, program_text: str
) -> dict:
    system = "You are a knowledgeable research assistant."
    if program_text:
        system += f"\n\nResearch context:\n{program_text[:500]}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "context": context[:512],
    }


def _write_jsonl(records: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="autoresearch prepare step")
    parser.add_argument("--corpus", default="data/all_docs_cleaned.txt")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--program", default="templates/program.md")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--use-llm", action="store_true",
                        help="Shorthand for --source-type llm")
    parser.add_argument("--model", default=None,
                        help="LLM model: gpt-4o-mini | claude-* | ollama:llama3.2 (defaults to best available)")
    parser.add_argument("--source-type", default="heuristic",
                        choices=["heuristic", "llm", "notebooklm"],
                        help="Q&A backend: notebooklm > llm > heuristic (auto-fallback)")
    parser.add_argument("--pdf-sources", nargs="*", default=None,
                        help="PDF paths to upload to NotebookLM (--source-type notebooklm)")
    args = parser.parse_args()

    prepare(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        program_md=args.program,
        max_pairs=args.max_pairs,
        val_frac=args.val_frac,
        use_llm=args.use_llm,
        llm_model=args.model,
        source_type=args.source_type,
        pdf_sources=args.pdf_sources,
    )
