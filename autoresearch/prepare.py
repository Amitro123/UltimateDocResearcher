"""
autoresearch/prepare.py
-----------------------
Wraps (and extends) karpathy/autoresearch's data preparation step.

Takes data/all_docs_cleaned.txt and transforms it into:
  • data/train.jsonl  — (question, answer) pairs for supervised fine-tuning
  • data/val.jsonl    — held-out evaluation set
  • data/program.md   — research program injected into the model

The question-generation loop uses a small LLM (via OpenAI API or
a local model) to synthesise Q&A pairs from each chunk — matching
autoresearch's self-generating training data philosophy.
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
    Call an OpenAI-compatible API to generate a Q&A pair from a passage.
    Returns (question, answer) or None on failure.
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=api_base or os.getenv("OPENAI_API_BASE", None),
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Passage:\n{passage[:3000]}"},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        text = resp.choices[0].message.content.strip()
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
):
    """
    Full prepare pipeline.

    Args:
        corpus_path: cleaned corpus from UltimateCollector/analyzer
        output_dir:  where to write train/val jsonl
        program_md:  research program template (injected into prompts)
        val_frac:    fraction held out for validation
        max_pairs:   maximum total Q&A pairs to generate
        use_llm:     if True, use OpenAI API for richer Q&A; else heuristic
        llm_model:   model name for LLM Q&A generation
        seed:        random seed for reproducibility
    """
    random.seed(seed)
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    # Load program context
    program_text = ""
    if Path(program_md).exists():
        program_text = Path(program_md).read_text(encoding="utf-8")

    # Load and split corpus into chunks
    raw = corpus_path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in re.split(r"\n\n+", raw) if len(c.strip()) > 150]
    print(f"[prepare] {len(chunks)} chunks loaded from {corpus_path}")

    # Generate Q&A pairs
    all_pairs: List[dict] = []
    random.shuffle(chunks)

    for i, chunk in enumerate(chunks):
        if len(all_pairs) >= max_pairs:
            break
        if i % 50 == 0:
            print(f"  Processing chunk {i}/{len(chunks)}…", flush=True)

        if use_llm:
            result = generate_qa_pair(chunk, model=llm_model)
            if result:
                q, a = result
                all_pairs.append(_make_record(q, a, chunk, program_text))
        else:
            for q, a in heuristic_qa(chunk):
                if len(all_pairs) >= max_pairs:
                    break
                all_pairs.append(_make_record(q, a, chunk, program_text))

    print(f"[prepare] Generated {len(all_pairs)} Q&A pairs")

    # Train / val split
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
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    prepare(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        program_md=args.program,
        max_pairs=args.max_pairs,
        val_frac=args.val_frac,
        use_llm=args.use_llm,
        llm_model=args.model,
    )
