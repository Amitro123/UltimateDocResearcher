# CLAUDE.md — project-rules-generator

> Operational guidelines and rules for working with UltimateDocResearcher,
> derived from the E2E verification on 2026-03-14; updated 2026-03-15.

## Build & Run Commands

| Goal | Command |
|------|---------|
| **Start a new topic** (run this first!) | `python new_run.py --topic "your new topic"` |
| Check LLM Setup | `python -m autoresearch.llm_client check` |
| Full Research Loop | `python autoresearch/train.py --topic "topic" --iterations 1` |
| Collect Documents | `python -m collector.ultimate_collector --pdf-dir papers/ --github user/repo --output-dir data/` |
| Analyze Corpus | `python -c "from collector.analyzer import analyze_corpus; analyze_corpus('data/all_docs.txt', 'data/')"` |
| Prepare Q&A | `python autoresearch/prepare.py --corpus data/all_docs_cleaned.txt --max-pairs 50` |
| Code Suggestions | `python -m autoresearch.code_suggester --corpus data/all_docs_cleaned.txt --topic "topic" --output results/code_suggestions.md` |
| **Multi-format Research Package** | `python -m autoresearch.research --topic "topic"` |
| Run Eval | `python -m eval.run_eval --input results/code_suggestions.md --threshold 3.5` |

## Code Style & Patterns

- **Python**: Use async-first approach for I/O bound tasks (scrapers).
- **Graceful Failure**: Always include heuristic fallbacks for LLM-dependent functions.
- **Paths**: Use absolute paths or ensure directories exist (`papers/`, `data/`, `results/`) before writing.
- **Connectivity**: Local Ollama (at `http://localhost:11434`) is prone to connection errors. Implement retries with exponential backoff for all `llm_client` calls.
- **Reporting**: Save intermediate results to `data/` and final reports to `results/`. Use JSON for metrics and Markdown for human-readable output.

## Operational Rules

1. **New Topic Reset**: Always run `python new_run.py --topic "your topic"` before starting a new research topic. This archives `papers/`, clears stale `data/` artifacts, and updates `templates/program.md`. Skipping this step causes corpus contamination from previous runs.
2. **Pre-flight Check**: Always run `python -m autoresearch.llm_client check` before starting a research loop. The check now sends a hello-world chat to confirm the model actually responds, not just that the server is up.
3. **Directory Safety**: `--pdf-dir` and `--output-dir` are now auto-created by the collector CLI if missing. No manual `mkdir` needed.
4. **Heuristic Fallback**: If Ollama fails after retries, use `--source-type heuristic` in `prepare.py` and fall back to `_heuristic_suggestions` in `code_suggester.py`.
5. **Retries Built-in**: `llm_client` now retries Ollama/HTTP calls up to 3× with exponential back-off (1 s, 2 s, 4 s + jitter). You will see `⚠️ Retrying in Xs…` messages on transient failures — this is expected.
6. **Gemini Rate Limits**: Gemini free-tier keys hit per-minute quotas. `_chat_google()` retries up to 4× with 15 s → 30 s → 60 s → 120 s back-off. You will see `⚠️ Gemini rate limit` messages — this is expected and will resolve automatically. If it keeps failing, wait a few minutes before retrying.
7. **Timeouts**: Individual HTTP calls time out after 60 s (down from 120 s). If the LLM is unresponsive after all retries, a `RuntimeError` is raised with a clear message.
