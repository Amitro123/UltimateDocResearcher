#!/usr/bin/env bash
# quickstart/demo-run.sh
# ─────────────────────────────────────────────────────────────────────────────
# 2-minute demo of UltimateDocResearcher.
#
# What it does:
#   1. Checks Python and LLM availability
#   2. Resets workspace to a clean demo state
#   3. Loads the bundled demo corpus (no PDF download needed)
#   4. Runs code_suggester with heuristic fallback (works without any LLM)
#   5. Generates a multi-format research package
#   6. Prints the output location
#
# Usage:
#   bash quickstart/demo-run.sh
#   make quickstart

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from project root

DEMO_TOPIC="Claude tool use and SDK patterns"
DEMO_CORPUS="quickstart/demo-corpus.txt"
DEMO_OUTPUT="results/quickstart-demo"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RESET="\033[0m"
BOLD="\033[1m"

step() { echo -e "\n${CYAN}▶ $*${RESET}"; }
ok()   { echo -e "  ${GREEN}✅  $*${RESET}"; }
warn() { echo -e "  ${YELLOW}⚠️   $*${RESET}"; }

# ── 1. Python check ───────────────────────────────────────────────────────────
step "Checking Python…"
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PYVER found"

# ── 2. Dependency check (non-fatal) ──────────────────────────────────────────
step "Checking dependencies…"
python3 -c "import jinja2, yaml, streamlit" 2>/dev/null && \
    ok "Core dependencies present" || \
    warn "Some dependencies missing. Run: pip install -r requirements.txt"

# ── 3. LLM check ─────────────────────────────────────────────────────────────
step "Detecting LLM backend…"
LLM_MODEL=$(python3 -m autoresearch.llm_client auto 2>/dev/null | tail -1 || echo "heuristic")
if [ "$LLM_MODEL" = "heuristic" ]; then
    warn "No LLM detected — will use heuristic fallback (lower quality, but works offline)"
    warn "For better results: install Ollama (https://ollama.com) or set GOOGLE_API_KEY"
else
    ok "LLM: $LLM_MODEL"
fi

# ── 4. Build demo corpus if needed ───────────────────────────────────────────
step "Preparing demo corpus…"
if [ ! -f "$DEMO_CORPUS" ]; then
    python3 quickstart/build_demo_corpus.py
fi
ok "Demo corpus ready ($(wc -c < "$DEMO_CORPUS" | tr -d ' ') bytes)"

# ── 5. Workspace reset ────────────────────────────────────────────────────────
step "Resetting workspace for demo topic…"
python3 new_run.py --topic "$DEMO_TOPIC" --skip-archive --keep-cache --dry-run > /dev/null 2>&1 || true
# Copy demo corpus into data/ without a full reset
mkdir -p data
cp "$DEMO_CORPUS" data/all_docs_cleaned.txt
# Create a minimal external_docs.txt so the weighted sampler works
cp "$DEMO_CORPUS" data/external_docs.txt
ok "Workspace ready"

# ── 6. Generate code suggestions ─────────────────────────────────────────────
step "Generating code suggestions…"
mkdir -p results
python3 -m autoresearch.code_suggester \
    --corpus data/all_docs_cleaned.txt \
    --topic "$DEMO_TOPIC" \
    --n-suggestions 3 \
    --output results/quickstart-code-suggestions.md 2>&1 | grep -v "^$" || true
ok "Code suggestions → results/quickstart-code-suggestions.md"

# ── 7. Generate multi-format package ─────────────────────────────────────────
step "Generating research package…"
python3 -m autoresearch.research \
    --topic "$DEMO_TOPIC" \
    --corpus data/all_docs_cleaned.txt \
    --output-dir "$DEMO_OUTPUT" \
    --run-id quickstart-demo \
    --no-code 2>&1 | grep -E "^\[|✅|❌" || true
ok "Research package → $DEMO_OUTPUT/"

# ── 8. Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║            ✅  Demo complete!                            ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${GREEN}Code suggestions:${RESET}  results/quickstart-code-suggestions.md"
echo ""
echo -e "  ${GREEN}Research package:${RESET}  $DEMO_OUTPUT/"
for f in SUMMARY.md IMPLEMENTATION.md NEXT_STEPS.md; do
    [ -f "$DEMO_OUTPUT/$f" ] && echo "    ✅  $f" || echo "    —   $f (not generated)"
done
echo ""
echo -e "  ${CYAN}Next steps:${RESET}"
echo "    1. Read results/quickstart-code-suggestions.md"
echo "    2. Launch the dashboard: make dashboard"
echo "    3. Run on your own topic:"
echo "       python new_run.py --topic \"your topic\""
echo "       python -m autoresearch.research --topic \"your topic\""
echo ""
