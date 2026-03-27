# Contributing to UltimateDocResearcher

Thank you for your interest in contributing! This guide covers how to set up your
development environment, run tests, and submit pull requests.

---

## Dev Environment Setup

```bash
# 1. Clone the repo
git clone https://github.com/Amitro123/UltimateDocResearcher
cd ultimate-doc-researcher

# 2. Create and activate a virtual environment (Python 3.11+)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Copy the example env file and fill in any keys you want to use
cp .env.example .env
```

You can run the pipeline with **no API keys** using the heuristic fallback — you only
need keys to use cloud LLMs (Gemini, OpenAI, Anthropic) or Kaggle remote training.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_scraper.py -v

# With coverage (requires pytest-cov)
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

All 347 tests must pass with zero warnings before a PR is merged.

---

## Running the E2E Pipeline

Before submitting changes to the pipeline, verify the full flow works:

```bash
# Reset workspace (required before each new topic)
python new_run.py --topic "test topic"

# Collect a small corpus
python -m collector.ultimate_collector \
  --queries "test query" \
  --output-dir data/

# Analyze
python -c "from collector.analyzer import analyze_corpus; analyze_corpus('data/all_docs.txt', 'data/')"

# Prepare Q&A (heuristic — no API key needed)
python autoresearch/prepare.py --corpus data/all_docs_cleaned.txt --max-pairs 20

# Generate code suggestions
python -m autoresearch.code_suggester \
  --corpus data/all_docs_cleaned.txt \
  --model heuristic \
  --n-suggestions 3
```

See [`E2E_GUIDE.md`](E2E_GUIDE.md) for the full walkthrough.

---

## Code Style

- **Python 3.11+**
- Formatter: [ruff](https://docs.astral.sh/ruff/) — `ruff format .`
- Linter: `ruff check .`
- Max line length: 100 characters (configured in `pyproject.toml`)
- Async-first for all I/O-bound code (see `collector/scraper.py` for examples)
- Always include heuristic fallbacks for LLM-dependent functions
- Use `encoding="utf-8"` on all `open()` and `read_text()` calls

---

## Pull Request Conventions

1. **Branch name**: `fix/<short-description>` or `feat/<short-description>`
2. **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) style:
   - `fix: correct asyncio event loop handling in scraper`
   - `feat: add incremental collect metadata tracking`
   - `docs: update E2E_GUIDE with NotebookLM instructions`
3. **Tests**: Add or update tests for every bug fix and new feature
4. **No secrets**: Never commit `.env`, API keys, or credentials — they are `.gitignore`d
5. **PR description**: Describe what changed, why, and how to test it

---

## Project Structure

See [`AGENTS.md`](AGENTS.md) for the full architecture overview and phase-by-phase
implementation history.

```
collector/          # Document collection (PDF, web, Drive, GitHub)
autoresearch/       # Q&A prep, training loop, eval, code suggestions
eval/               # Standardized 5-criteria eval framework
research_deliverables/  # Multi-format output generators
memory/             # SQLite run history + prompt cache
dashboard/          # Streamlit dashboard
tests/              # pytest test suite (347 tests)
```

---

## Reporting Issues

Open a GitHub issue with:
- Your OS + Python version
- The exact command you ran
- The full error message / stack trace
- Any relevant environment details (Ollama version, API key provider, etc.)
