# UltimateDocResearcher — developer Makefile
#
# Usage:
#   make quickstart      # 2-minute demo — no API key, no Ollama install needed
#   make dev             # install dev dependencies in current venv
#   make test            # run all tests
#   make docker-up       # start full Docker stack
#   make docker-down     # stop Docker stack
#   make check           # verify LLM backend is reachable
#   make clean           # remove generated data/results artifacts

.DEFAULT_GOAL := help
PYTHON        := python3
PIP           := pip3
COMPOSE       := docker compose -f docker/docker-compose.yml
TOPIC         ?= "Claude tool use patterns"   # override: make research TOPIC="my topic"

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "  UltimateDocResearcher"
	@echo ""
	@echo "  make quickstart    2-minute demo with sample corpus (no API key needed)"
	@echo "  make dev           install dev dependencies"
	@echo "  make test          run test suite"
	@echo "  make check         verify LLM backend is reachable"
	@echo "  make research      run pipeline on TOPIC= (default: Claude tool use)"
	@echo "  make dashboard     launch Streamlit dashboard"
	@echo "  make chat          launch Streamlit chat interface"
	@echo "  make docker-up     start full Docker stack (Ollama + dashboard + chat)"
	@echo "  make docker-down   stop Docker stack"
	@echo "  make clean         remove data/results artifacts (keeps papers/)"
	@echo ""

# ── Quickstart ────────────────────────────────────────────────────────────────

.PHONY: quickstart
quickstart: _check-python _install-core
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║      UltimateDocResearcher — 2-minute quickstart         ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@bash quickstart/demo-run.sh

# ── Development ───────────────────────────────────────────────────────────────

.PHONY: dev
dev: _check-python
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio
	@echo ""
	@echo "✅  Dev dependencies installed."
	@echo "    Run 'make check' to verify your LLM backend."
	@echo ""

.PHONY: _install-core
_install-core:
	@$(PIP) install -q -r requirements.txt

# ── Tests ─────────────────────────────────────────────────────────────────────

.PHONY: test
test: _check-python
	@echo "Running test suite…"
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo ""
	@echo "Running classifier smoke test…"
	$(PYTHON) -c "\
from research_deliverables.classify_topic import classify_topic; \
types = [classify_topic(t).research_type for t in [ \
    'Claude SDK patterns', 'streaming pipeline architecture', \
    'fine-tuning with RLHF', 'survey of eval frameworks']]; \
assert types == ['code','arch','process','market'], types; \
print('✅  classify_topic: all 4 types correct')"
	@echo ""
	@echo "✅  All tests passed."

# ── LLM check ─────────────────────────────────────────────────────────────────

.PHONY: check
check:
	$(PYTHON) -m autoresearch.llm_client check
	$(PYTHON) -m autoresearch.llm_client auto

# ── Research pipeline ─────────────────────────────────────────────────────────

.PHONY: research
research:
	@echo "Running research pipeline for topic: $(TOPIC)"
	$(PYTHON) new_run.py --topic $(TOPIC)
	$(PYTHON) -m autoresearch.research --topic $(TOPIC)

# ── Dashboard / Chat ──────────────────────────────────────────────────────────

.PHONY: dashboard
dashboard:
	@echo "Opening dashboard → http://localhost:8501"
	streamlit run dashboard/app.py --server.port 8501

.PHONY: chat
chat:
	@echo "Opening chat → http://localhost:8503"
	streamlit run chat/app.py --server.port 8503

# ── Docker ────────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up:
	@echo "Starting Docker stack…"
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; fi
	$(COMPOSE) up --build -d
	@echo ""
	@echo "  Dashboard → http://localhost:8501"
	@echo "  Chat      → http://localhost:8503"
	@echo ""
	@echo "First time? Pull the LLM model (one-time, ~2 GB):"
	@echo "  $(COMPOSE) run --rm ollama-pull"
	@echo ""

.PHONY: docker-down
docker-down:
	$(COMPOSE) down

.PHONY: docker-logs
docker-logs:
	$(COMPOSE) logs -f

.PHONY: docker-shell
docker-shell:
	$(COMPOSE) run --rm app bash

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	@echo "Removing generated artifacts (data/, results/) — keeping papers/ and memory/…"
	@rm -f data/all_docs*.txt data/external_docs.txt data/corpus_report.json
	@rm -f data/train.jsonl data/val.jsonl
	@echo "✅  Clean done. Run 'make quickstart' to regenerate demo data."

.PHONY: clean-all
clean-all: clean
	@echo "Full reset: removing results/, memory/prompts.db, memory/runs.db…"
	@rm -rf results/
	@rm -f memory/prompts.db memory/runs.db
	@echo "✅  Full reset done."

# ── Internal checks ───────────────────────────────────────────────────────────

.PHONY: _check-python
_check-python:
	@$(PYTHON) --version > /dev/null 2>&1 || (echo "❌  python3 not found. Install Python 3.10+." && exit 1)
	@$(PYTHON) -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" || \
		(echo "❌  Python 3.10+ required." && exit 1)
