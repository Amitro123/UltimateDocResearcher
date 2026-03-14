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
from datetime import datetime
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
    metrics["timestamp"] = datetime.utcnow().isoformat()

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
    """Append metrics row to results.tsv, creating headers if needed."""
    path = Path(tsv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
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
):
    """
    Full autoresearch loop:
      for i in range(n_iterations):
        1. Collect (or re-collect with expanded queries)
        2. Prepare (generate Q&A)
        3. Train (LoRA fine-tune)
        4. Evaluate → append to results.tsv
        5. Commit results to git
    """
    cfg = config or TrainConfig(topic=topic)

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

        # 4. Optionally commit results
        _git_commit_results(i + 1, metrics)

        print(f"  val_score = {metrics.get('val_score', 'N/A')}")


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
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model,
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        results_tsv=args.results_tsv,
        topic=args.topic,
    )
    train(cfg)
