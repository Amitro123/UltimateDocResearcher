"""
api-triggers/poll_results.py
-----------------------------
Polls a running Kaggle kernel and downloads results when complete.
Also handles syncing results.tsv back to the GitHub repo.

Can be run locally or as a GitHub Actions job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def sync_results_to_git(
    results_dir: str = "results/",
    remote: str = "origin",
    branch: str = "main",
) -> None:
    """Pull latest, add results/, commit, push."""
    cmds = [
        ["git", "pull", remote, branch, "--rebase"],
        ["git", "add", results_dir],
        ["git", "commit", "-m", "auto: sync results from Kaggle"],
        ["git", "push", remote, branch],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[git] {' '.join(cmd)} → {result.stderr.strip()}", file=sys.stderr)
        else:
            print(f"[git] ✅ {' '.join(cmd)}")


def print_results_summary(tsv_path: str = "results/results.tsv") -> None:
    """Print a quick summary of results.tsv."""
    path = Path(tsv_path)
    if not path.exists():
        print(f"[results] No results.tsv at {tsv_path}")
        return
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        print("[results] No data rows yet")
        return
    headers = lines[0].split("\t")
    print(f"\n{'─'*60}")
    print(f"  Results: {len(lines)-1} iterations")
    print(f"{'─'*60}")
    for i, line in enumerate(lines[1:], 1):
        row = dict(zip(headers, line.split("\t")))
        print(
            f"  iter {row.get('iteration','?'):>3}  "
            f"val_loss={row.get('val_loss','?'):>8}  "
            f"val_score={row.get('val_score','?'):>6}  "
            f"topic={row.get('topic','?')}"
        )
    print(f"{'─'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, help="Kaggle kernel slug (user/kernel-id)")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in minutes")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--git-sync", action="store_true")
    args = parser.parse_args()

    from trigger_kaggle import KaggleRunner
    runner = KaggleRunner()

    status = runner.poll_until_done(
        args.slug,
        poll_interval=args.poll_interval,
        timeout_minutes=args.timeout,
    )

    if status == "complete":
        runner.download_output(args.slug, args.results_dir)
        print_results_summary(f"{args.results_dir}/results.tsv")
        if args.git_sync:
            sync_results_to_git(args.results_dir)
    else:
        print(f"[poll] Kernel ended with status: {status}")
        sys.exit(1)
