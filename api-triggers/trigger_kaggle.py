"""
api-triggers/trigger_kaggle.py
-------------------------------
Pushes a kernel to Kaggle via the Kaggle API and monitors it.

Usage:
    python trigger_kaggle.py --topic "claude skills optimization" --iterations 20
    python trigger_kaggle.py --kernel my-username/ultimate-doc-researcher --poll

Environment variables required:
    KAGGLE_USERNAME
    KAGGLE_KEY
    GITHUB_REPO        (optional, for git-commit-back step)
    GITHUB_TOKEN       (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ── Kaggle API wrapper ────────────────────────────────────────────────────────

class KaggleRunner:
    def __init__(self, username: Optional[str] = None, key: Optional[str] = None):
        self.username = username or os.environ.get("KAGGLE_USERNAME")
        self.key = key or os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")
        
        if not self.username or not self.key:
            raise ValueError("Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY/KAGGLE_API_TOKEN.")
        # Kaggle SDK reads from ~/.kaggle/kaggle.json
        self._ensure_credentials()

    def _ensure_credentials(self) -> None:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        creds_path = kaggle_dir / "kaggle.json"
        if not creds_path.exists():
            creds_path.write_text(
                json.dumps({"username": self.username, "key": self.key})
            )
            creds_path.chmod(0o600)

    def _run(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["kaggle", *args],
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            msg = f"kaggle {' '.join(args)} failed (exit {result.returncode})"
            if result.stdout: msg += f"\nSTDOUT: {result.stdout}"
            if result.stderr: msg += f"\nSTDERR: {result.stderr}"
            raise RuntimeError(msg)
        return result.stdout.strip()

    def push_kernel(self, kernel_dir: str | Path) -> str:
        """
        Push kernel from directory containing kernel-metadata.json.
        Returns the kernel slug (username/kernel-id).
        """
        kernel_dir = Path(kernel_dir)
        meta_path = kernel_dir / "kernel-metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"kernel-metadata.json not found in {kernel_dir}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_id = meta['id']
        
        # Avoid double username prefix if metadata already has it
        if "/" in meta_id:
            slug = meta_id
        else:
            slug = f"{self.username}/{meta_id}"

        print(f"[kaggle] Pushing kernel: {slug}")
        output = self._run("kernels", "push", "-p", str(kernel_dir))
        print(f"[kaggle] Push response: {output}")
        return slug

    def get_status(self, slug: str) -> str:
        """Return kernel run status: 'running' | 'complete' | 'error'"""
        output = self._run("kernels", "status", slug, check=False)
        for line in output.lower().split("\n"):
            if "running" in line:
                return "running"
            if "complete" in line:
                return "complete"
            if "error" in line or "fail" in line:
                return "error"
        return "unknown"

    def poll_until_done(
        self,
        slug: str,
        poll_interval: int = 60,
        timeout_minutes: int = 120,
    ) -> str:
        """Poll status every `poll_interval` seconds until done or timeout."""
        print(f"[kaggle] Polling {slug} (every {poll_interval}s, timeout {timeout_minutes}m)")
        start = time.time()
        timeout = timeout_minutes * 60
        while True:
            status = self.get_status(slug)
            elapsed = (time.time() - start) / 60
            print(f"  [{elapsed:.1f}m] status = {status}")
            if status in ("complete", "error"):
                return status
            if time.time() - start > timeout:
                print("[kaggle] Timeout reached")
                return "timeout"
            time.sleep(poll_interval)

    def download_output(self, slug: str, dest: str | Path = "results/") -> Path:
        """Download kernel output files."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self._run("kernels", "output", slug, "-p", str(dest))
        print(f"[kaggle] Output downloaded → {dest}")
        return dest

    def list_output_files(self, slug: str) -> list[str]:
        output = self._run("kernels", "output", slug, "--list", check=False)
        return [line.strip() for line in output.split("\n") if line.strip()]


# ── Notebook generator ────────────────────────────────────────────────────────

def generate_kernel_notebook(
    topic: str,
    n_iterations: int,
    github_repo: str,
    output_path: str | Path,
) -> Path:
    """
    Generate a Kaggle notebook (.ipynb) that:
      1. Clones the GitHub repo
      2. Installs dependencies
      3. Runs the full collect → prepare → train loop
    """
    cells = [
        _code_cell(f'# UltimateDocResearcher — {topic.replace(chr(10), " ")}'),
        _code_cell(f"""\
import subprocess, os, sys
print("🚀 Starting UltimateDocResearcher research loop...")

# 1. Clone & Setup
repo = {json.dumps(github_repo)}
print(f"Cloning repo: {{repo}}")

if not os.path.exists("ultimate-doc-researcher"):
    try:
        subprocess.run(["git", "clone", f"https://github.com/{{repo}}.git", "ultimate-doc-researcher"], check=True)
    except Exception as e:
        print(f"❌ Clone failed: {{e}}")
        # Try fallback if possible or exit
        raise

os.chdir("ultimate-doc-researcher")
sys.path.insert(0, ".")
print("✅ Repo cloned and directoy changed")
"""),
        _code_cell("""\
# 2. Install Dependencies
print("📦 Installing dependencies (this may take 2-3 minutes)...")
subprocess.run([
    "pip", "install", "-q", "--upgrade",
    "unsloth", "trl", "peft", "pymupdf",
    "aiohttp", "beautifulsoup4", "google-api-python-client", "google-auth",
    "bitsandbytes", "accelerate", "xformers",
], check=True)
print("✅ Dependencies installed")
"""),
        _code_cell(f"""\
# 3. Collect & Prepare
from collector.ultimate_collector import UltimateCollector
from autoresearch.prepare import prepare
from collector.analyzer import analyze_corpus

TOPIC = {json.dumps(topic)}
N_ITERATIONS = {int(n_iterations)}

print(f"🔍 Collecting documents for: {{TOPIC}}")
collector = UltimateCollector(
    google_queries=[f"{{TOPIC}} research paper", f"{{TOPIC}} tutorial"],
    reddit_subreddits=["MachineLearning", "LocalLLaMA"],
    output_dir="data/",
)
docs = collector.run()
print(f"✅ Collected {{len(docs)}} documents")

print("🧹 Analyzing and cleaning corpus...")
report = analyze_corpus("data/all_docs.txt", "data/")
print(report)

print("📝 Preparing Q&A pairs...")
prepare(
    corpus_path="data/all_docs_cleaned.txt",
    output_dir="data/",
    program_md="templates/program.md",
    max_pairs=500,
    use_llm=False,
)
print("✅ Data preparation complete")
"""),
        _code_cell(f"""\
# 4. Training Loop
from autoresearch.train import TrainConfig, train

cfg = TrainConfig(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    num_train_epochs=1,
    topic=TOPIC,
    output_dir="/kaggle/working/models/lora",
    results_tsv="/kaggle/working/results.tsv",
)

print(f"🏋️ Starting research loop ({{N_ITERATIONS}} iterations)...")
for i in range(N_ITERATIONS):
    cfg.iteration = i + 1
    metrics = train(cfg)
    vs = metrics.get('val_score', 'N/A')
    print(f"Iteration {{i+1}}/{{N_ITERATIONS}}: val_score={{vs}}")

print("✅ Training sequence complete")
"""),
        _code_cell("""\
# 5. Export Results
import pandas as pd
if os.path.exists("/kaggle/working/results.tsv"):
    df = pd.read_csv("/kaggle/working/results.tsv", sep="\\t")
    print(df[["iteration", "val_loss", "val_score", "elapsed_seconds"]])
    print("✅ Results exported")
else:
    print("⚠️ results.tsv not found")
""")
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2))
    print(f"[notebook] Generated: {output_path}")
    return output_path


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cell_{hash(source) & 0xFFFF:04x}",
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def slugify(text: str) -> str:
    """Convert text to a valid Kaggle ID slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


# ── kernel-metadata.json ──────────────────────────────────────────────────────

def generate_kernel_metadata(
    kernel_id: str,
    title: str,
    notebook_path: str,
    username: str,
    output_path: str | Path,
    enable_gpu: bool = True,
    enable_internet: bool = True,
) -> Path:
    meta = {
        "id": f"{username}/{kernel_id}",
        "title": title,
        "code_file": notebook_path,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    output_path = Path(output_path)
    output_path.write_text(json.dumps(meta, indent=2))
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description="Trigger Kaggle kernel for autoresearch")
    parser.add_argument("--topic", default="Claude skills optimization")
    parser.add_argument("--iterations", type=int, default=20)
    
    # Try to detect current repo
    try:
        remote_url = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
        repo_path = remote_url.replace("https://github.com/", "").replace(".git", "")
    except Exception:
        repo_path = "yourusername/ultimate-doc-researcher"
        
    parser.add_argument("--github-repo", default=repo_path)
    parser.add_argument("--kernel-dir", default="templates/kaggle_kernel/")
    parser.add_argument("--poll", action="store_true", help="Only poll an existing kernel")
    parser.add_argument("--slug", default="", help="Kernel slug for --poll mode")
    parser.add_argument("--download-results", action="store_true")
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()

    runner = KaggleRunner()

    if args.poll:
        slug = args.slug or f"{runner.username}/ultimate-doc-researcher"
        status = runner.poll_until_done(slug)
        if status == "complete" and args.download_results:
            runner.download_output(slug, args.results_dir)
        return

    # Generate notebook + metadata
    kernel_dir = Path(args.kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    topic_slug = slugify(args.topic)
    title = f"UltimateDocResearcher — {args.topic[:50]}"
    # Append timestamp to avoid 409 Conflict on reused titles/ids
    kernel_id = slugify(f"doc-researcher-{topic_slug}-{int(time.time())}")

    print(f"[kaggle] Triggering run for topic: {args.topic}")
    print(f"[kaggle] Kernel title: {title}")
    print(f"[kaggle] Kernel ID: {kernel_id}")

    nb_path = generate_kernel_notebook(
        topic=args.topic,
        n_iterations=args.iterations,
        github_repo=args.github_repo,
        output_path=kernel_dir / "notebook.ipynb",
    )

    generate_kernel_metadata(
        kernel_id=kernel_id,
        title=title,
        notebook_path="notebook.ipynb",
        username=runner.username,
        output_path=kernel_dir / "kernel-metadata.json",
    )

    # Push
    slug = runner.push_kernel(kernel_dir)

    # Poll
    status = runner.poll_until_done(slug, poll_interval=60, timeout_minutes=120)
    print(f"\n[kaggle] Final status: {status}")

    if status == "complete" and args.download_results:
        runner.download_output(slug, args.results_dir)


if __name__ == "__main__":
    main()
