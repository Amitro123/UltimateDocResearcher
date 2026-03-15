"""
autoresearch/train.py
---------------------
Fine-tunes a language model on data/train.jsonl using LoRA / QLoRA.

Designed to run on Kaggle's free T4/P100 GPU in <2h for most configs.

Supports:
  • unsloth (2x faster, recommended on Kaggle)
  • HuggingFace PEFT + transformers (fallback)
  • Local CPU smoke-test mode (no GPU)

After training, evaluates on val.jsonl and appends to results/results.tsv.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_name: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # Training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    save_steps: int = 100
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = True
    optim: str = "adamw_8bit"
    output_dir: str = "models/lora_adapter"
    seed: int = 42

    # Data
    train_path: str = "data/train.jsonl"
    val_path: str = "data/val.jsonl"
    results_tsv: str = "results/results.tsv"

    # Run metadata
    iteration: int = 0
    topic: str = ""


# ── Training ──────────────────────────────────────────────────────────────────

def train(config: TrainConfig) -> dict:
    """
    Run training. Returns metrics dict.
    Auto-selects unsloth if available, falls back to HF transformers.
    """
    start = time.time()
    if _has_unsloth():
        metrics = _train_unsloth(config)
    else:
        metrics = _train_hf(config)

    elapsed = time.time() - start
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["iteration"] = config.iteration
    metrics["topic"] = config.topic
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

    _append_results(metrics, config.results_tsv)
    print(f"\n✅ Training complete in {elapsed/60:.1f}min  val_loss={metrics.get('val_loss', 'N/A')}")
    return metrics


def _has_unsloth() -> bool:
    try:
        import unsloth  # noqa
        return True
    except ImportError:
        return False


def _train_unsloth(config: TrainConfig) -> dict:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from datasets import load_dataset

    print(f"[train] Loading model (unsloth): {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules.split(","),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    train_dataset = _load_dataset(config.train_path, tokenizer)
    val_dataset = _load_dataset(config.val_path, tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        optim=config.optim,
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        seed=config.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        args=training_args,
    )
    trainer_stats = trainer.train()
    eval_results = trainer.evaluate()

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    return {
        "train_loss": round(trainer_stats.training_loss, 4),
        "val_loss": round(eval_results.get("eval_loss", float("nan")), 4),
        "val_score": _loss_to_score(eval_results.get("eval_loss", float("nan"))),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }


def _train_hf(config: TrainConfig) -> dict:
    """HuggingFace PEFT fallback — no unsloth."""
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"[train] Loading model (HF PEFT): {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules.split(","),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = _load_dataset(config.train_path, tokenizer)
    val_dataset = _load_dataset(config.val_path, tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        output_dir=config.output_dir,
        seed=config.seed,
        report_to="none",
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer_stats = trainer.train()
    eval_results = trainer.evaluate()

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    return {
        "train_loss": round(trainer_stats.training_loss, 4),
        "val_loss": round(eval_results.get("eval_loss", float("nan")), 4),
        "val_score": _loss_to_score(eval_results.get("eval_loss", float("nan"))),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }


def _load_dataset(jsonl_path: str, tokenizer):
    """Load JSONL chat records and tokenize into HF Dataset."""
    from datasets import Dataset

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec.get("messages", [])
            # Format as chat template text
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            records.append({"text": text})

    return Dataset.from_list(records)


def _loss_to_score(loss: float) -> float:
    """Normalise cross-entropy loss to a 0-1 score (higher = better)."""
    if math.isnan(loss) or math.isinf(loss):
        return 0.0
    return round(max(0.0, 1.0 - loss / 10.0), 4)


def _append_results(metrics: dict, tsv_path: str) -> None:
    """Append metrics row to results.tsv, creating headers if needed.

    A header is written on the first row even if the file already exists but is
    empty (e.g. a stale results.tsv from a previous run that was truncated).
    This prevents pandas.read_csv() failures when the TSV has data rows but no
    column names.
    """
    path = Path(tsv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"[train] Results appended → {tsv_path}")


# ── Autoresearch loop ─────────────────────────────────────────────────────────

def research_loop(
    topic: str,
    n_iterations: int = 20,
    config: Optional[TrainConfig] = None,
    collect_fn=None,
    prepare_fn=None,
    # Eval settings
    run_llm_eval: bool = True,
    judge_model: str = "gpt-4o-mini",
    eval_max_samples: int = 50,
    eval_pass_threshold: float = 3.0,
    # Code suggestion settings
    run_code_suggestions: bool = True,
    suggestion_model: str = "gpt-4o-mini",
    corpus_path: str = "data/all_docs_cleaned.txt",
    n_code_suggestions: int = 5,
    # Memory settings
    run_id: Optional[int] = None,
    similarity_threshold: float = 0.8,
    skip_if_similar: bool = False,
):
    """
    Full autoresearch loop:
      for i in range(n_iterations):
        1. Collect (or re-collect with expanded queries)
        2. Prepare (generate Q&A)
        3. Train (LoRA fine-tune)
        4. LLM-as-Judge eval → results/eval_report.json
        5. Code suggestions → results/code_suggestions.md
        6. Commit results to git

    Args:
        run_llm_eval:        enable LLM-as-Judge evaluation after each iteration
        judge_model:         LLM to use as judge (gpt-4o-mini, claude-3-5-haiku-20241022, …)
        eval_max_samples:    cap on val samples to judge per iteration
        eval_pass_threshold: minimum overall score to "pass" a sample (1–5)
        run_code_suggestions: generate code suggestions after the LAST iteration
        suggestion_model:    LLM for code suggestion generation
        corpus_path:         path to cleaned corpus used for suggestions
        n_code_suggestions:  number of code snippets to generate
        run_id:              existing RunMemory id (passed from dashboard); creates one if None
        similarity_threshold: threshold for similar-topic check (0–1)
        skip_if_similar:     if True and a similar run exists, return early with its results
    """
    cfg = config or TrainConfig(topic=topic)

    # ── Memory: check for similar past runs ───────────────────────────────
    _mem = None
    try:
        from memory.memory import RunMemory
        _mem = RunMemory()

        similar = _mem.find_similar(topic, threshold=similarity_threshold)
        if similar and skip_if_similar:
            best = similar[0]
            print(
                f"\n[memory] Similar run found: '{best['topic']}' "
                f"(similarity={best['similarity']:.0%}, score={best.get('avg_score') or '?'})\n"
                f"[memory] Skipping research — use --no-skip-similar to force a new run."
            )
            return

        if similar:
            print(
                f"\n[memory] ℹ️  Similar past run: '{similar[0]['topic']}' "
                f"(similarity={similar[0]['similarity']:.0%}). "
                f"Starting new run anyway (use --skip-if-similar to reuse)."
            )

        # Register or reuse a run_id in the DB
        if run_id is None:
            run_id = _mem.start_run(topic, judge_model=judge_model)
            print(f"[memory] Run #{run_id} started → dashboard/runs.db")

    except Exception as exc:
        print(f"[memory] Memory system unavailable: {exc}", file=sys.stderr)
        _mem = None

    all_metrics: list[dict] = []

    for i in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"  ITERATION {i+1}/{n_iterations}  —  {topic}")
        print(f"{'='*60}")

        cfg.iteration = i + 1

        # 1. Collect
        if collect_fn:
            collect_fn(iteration=i)

        # 2. Prepare
        if prepare_fn:
            prepare_fn(iteration=i)

        # 3. Train
        metrics = train(cfg)

        # 4. LLM-as-Judge eval
        if run_llm_eval:
            try:
                from autoresearch.eval import run_eval
                eval_report = run_eval(
                    val_path=cfg.val_path,
                    model_path=cfg.output_dir if Path(cfg.output_dir).exists() else None,
                    judge_model=judge_model,
                    max_samples=eval_max_samples,
                    pass_threshold=eval_pass_threshold,
                    output_dir=Path(cfg.results_tsv).parent,
                    iteration=cfg.iteration,
                    topic=topic,
                )
                if eval_report and "summary" in eval_report:
                    metrics["judge_pass_rate"] = eval_report["summary"]["pass_rate"]
                    metrics["judge_avg_score"] = eval_report["summary"]["avg_overall"]
            except Exception as exc:
                print(f"[train] LLM eval skipped: {exc}", file=sys.stderr)

        all_metrics.append(metrics)

        # Log iteration to memory DB
        if _mem and run_id:
            try:
                _mem.log_iteration(
                    run_id, i + 1,
                    train_loss=metrics.get("train_loss"),
                    val_loss=metrics.get("val_loss"),
                    val_score=metrics.get("val_score"),
                    judge_pass_rate=metrics.get("judge_pass_rate"),
                    judge_avg_score=metrics.get("judge_avg_score"),
                )
            except Exception as exc:
                print(f"[memory] log_iteration failed: {exc}", file=sys.stderr)

        # 5. Code suggestions — run on the LAST iteration only
        is_last = (i + 1 == n_iterations)
        if run_code_suggestions and is_last:
            try:
                from autoresearch.code_suggester import generate_suggestions
                suggestions_path = Path(cfg.results_tsv).parent / "code_suggestions.md"
                generate_suggestions(
                    corpus_path=corpus_path,
                    topic=topic,
                    model=suggestion_model,
                    output_path=suggestions_path,
                    n_suggestions=n_code_suggestions,
                )
            except Exception as exc:
                print(f"[train] Code suggestions skipped: {exc}", file=sys.stderr)

        # 6. Optionally commit results
        _git_commit_results(i + 1, metrics)

        print(f"  val_score = {metrics.get('val_score', 'N/A')}")
        if "judge_avg_score" in metrics:
            print(f"  judge_avg_score = {metrics['judge_avg_score']} "
                  f"(pass_rate={metrics.get('judge_pass_rate', '?')})")

    # ── Memory: mark run complete ─────────────────────────────────────────
    if _mem and run_id and all_metrics:
        try:
            scores = [m["val_score"] for m in all_metrics if m.get("val_score") is not None]
            pass_rates = [m["judge_pass_rate"] for m in all_metrics if m.get("judge_pass_rate") is not None]
            suggestions_path = str(Path(cfg.results_tsv).parent / "code_suggestions.md")
            _mem.finish_run(
                run_id,
                status="completed",
                iterations=n_iterations,
                avg_score=round(sum(scores) / len(scores), 4) if scores else None,
                pass_rate=round(sum(pass_rates) / len(pass_rates), 4) if pass_rates else None,
                corpus_chars=Path(corpus_path).stat().st_size if Path(corpus_path).exists() else None,
                n_suggestions=n_code_suggestions,
                results_path=cfg.results_tsv,
                eval_path=str(Path(cfg.results_tsv).parent / "eval_report.json"),
                suggestions_path=suggestions_path,
            )
            print(f"\n[memory] Run #{run_id} completed → dashboard/runs.db")
        except Exception as exc:
            print(f"[memory] finish_run failed: {exc}", file=sys.stderr)
        finally:
            _mem.close()


def _git_commit_results(iteration: int, metrics: dict) -> None:
    import subprocess
    try:
        subprocess.run(
            ["git", "add", "results/"],
            check=True, capture_output=True
        )
        msg = (
            f"results: iter {iteration} "
            f"val_score={metrics.get('val_score', '?')} "
            f"val_loss={metrics.get('val_loss', '?')}"
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            check=True, capture_output=True
        )
        subprocess.run(
            ["git", "push"],
            check=True, capture_output=True
        )
        print(f"[train] Git commit: {msg}")
    except Exception as exc:
        print(f"[train] Git push skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--topic", default="")
    parser.add_argument("--output-dir", default="models/lora_adapter")
    parser.add_argument("--results-tsv", default="results/results.tsv")
    # Eval flags
    parser.add_argument("--no-eval", action="store_true", help="Skip LLM-as-Judge eval")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="Judge LLM (gpt-4o-mini, claude-3-5-haiku-20241022, …)")
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--eval-threshold", type=float, default=3.0)
    # Code suggestion flags
    parser.add_argument("--no-suggestions", action="store_true", help="Skip code suggestions")
    parser.add_argument("--suggestion-model", default="gpt-4o-mini")
    parser.add_argument("--corpus", default="data/all_docs_cleaned.txt")
    parser.add_argument("--n-suggestions", type=int, default=5)
    # Memory flags
    parser.add_argument("--run-id", type=int, default=None,
                        help="Existing RunMemory run id (set by dashboard)")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                        help="Topic similarity threshold for 'already researched' check")
    parser.add_argument("--skip-if-similar", action="store_true",
                        help="Exit early if a similar past run is found")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model,
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        results_tsv=args.results_tsv,
        topic=args.topic,
    )

    if args.iterations > 1:
        research_loop(
            topic=args.topic,
            n_iterations=args.iterations,
            config=cfg,
            run_llm_eval=not args.no_eval,
            judge_model=args.judge_model,
            eval_max_samples=args.eval_samples,
            eval_pass_threshold=args.eval_threshold,
            run_code_suggestions=not args.no_suggestions,
            suggestion_model=args.suggestion_model,
            corpus_path=args.corpus,
            n_code_suggestions=args.n_suggestions,
            run_id=args.run_id,
            similarity_threshold=args.similarity_threshold,
            skip_if_similar=args.skip_if_similar,
        )
    else:
        train(cfg)
