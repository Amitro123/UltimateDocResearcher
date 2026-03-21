# UltimateDocResearcher — End-to-End Guide

> How to go from "I want to improve X" to working code suggestions,
> using your own project as the research subject.

---

## The Big Picture

```
Your question / project
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 13 one-command entry (NEW)       │
│  python -m autoresearch.cli             │
│    --topic "X" --full --incremental     │
│                                         │
│  Runs steps 1–5 automatically.          │
│  Use --incremental to skip unchanged    │
│  sources on re-runs (fast).             │
└────────────────┬────────────────────────┘
                 │  (or run steps manually below)
        ▼
  1. COLLECT       ← PDFs, URLs, GitHub repos, your own codebase
        │             ↳ --incremental: skip unchanged sources
        │               (tracks hashes in data/collect_metadata.jsonl)
        ▼
  2. ANALYZE       ← quality filter, dedup, chunking
        │             writes external_docs.txt (70% of LLM window)
        ▼
  3. PREPARE       ← Q&A generation  (NotebookLM → LLM → heuristic)
        │
        ▼
  4. EVAL (Q&A)    ← LLM-as-Judge scores your Q&A pair quality
        │
        ▼
  5. SUGGEST       ← code snippets derived from your research
        │                        ↑ prompt cache (prompts.db / SQLite)
        ▼
  6. EVAL (output) ← 6-criteria score on code_suggestions.md
        │
        ├─── results/eval-report.json  +  results/code_suggestions.md
        │
        ▼
  5alt. RESEARCH PACKAGE  ← multi-format deliverables for any topic type
        │     python -m autoresearch.research --topic "..."
        ▼
  results/<run-id>/
    SUMMARY.md  ARCHITECTURE.md  IMPLEMENTATION.md
    RISKS.md    BENCHMARKS.md    NEXT_STEPS.md
    CODE/code_suggestions.md     metadata.json
        │
        ▼
  7. DASHBOARD     ← run history, metrics, Research Packages tab
                     (dashboard/app.py — New Run tab has incremental toggle)
```

Steps 4–6 run without a GPU. Step 3 is free with Ollama.
Only step "Train" (optional) needs Kaggle/Modal.

---

## Prerequisites

```bash
# Clone the repo
git clone https://github.com/Amitro123/UltimateDocResearcher
cd ultimate-doc-researcher
pip install -r requirements.txt
pip install pdfplumber            # for PDF extraction

# Choose your LLM backend (pick one):

# Option A — Free, local (recommended for dev)
# Install Ollama: https://ollama.com/download
ollama pull llama3.2

# Option B — Anthropic (best quality)
export ANTHROPIC_API_KEY=sk-ant-...

# Option C — OpenAI
export OPENAI_API_KEY=sk-...

# Option D — Google Gemini (free tier available)
pip install google-generativeai
export GOOGLE_API_KEY=AIza...
# Free-tier keys are rate-limited per minute. The client retries automatically
# (15 s → 30 s → 60 s → 120 s). If you hit sustained 429s, wait a few minutes.

# Option E — NotebookLM (best for PDFs, needs browser auth once)
pip install "notebooklm-py[browser]"
playwright install chromium
notebooklm login                  # opens browser once to authenticate
```

Check everything is wired:
```bash
python -m autoresearch.llm_client check      # is Ollama reachable?
python -m autoresearch.llm_client auto       # what's the best model available?
```

---

## Worked Example: Improving `project-rules-generator`

This is a real simulation of using UltimateDocResearcher to research
how to improve a Claude skills creator / project rules generator.

### Step 0b — Phase 13: one-command pipeline (skip to here for speed)

If you want everything in a single command instead of running steps 1–5
manually, use the Phase 13 CLI:

```bash
# Full pipeline: collect (incremental) + analyze + prepare + package
python -m autoresearch.cli \
  --topic "Improving project-rules-generator skill for Claude" \
  --full \
  --incremental \
  --pdf-dir papers/
```

`--incremental` reads `data/collect_metadata.jsonl` and skips any local
files or URLs whose hash hasn't changed since the last run — dramatically
faster on repeated research iterations.

```bash
# Force re-collect everything (ignore the incremental cache):
python -m autoresearch.cli --topic "..." --full --incremental --force-recollect

# Deliverables only (corpus already collected):
python -m autoresearch.cli --topic "..."

# Individual steps:
python -m autoresearch.cli --topic "..." --collect --incremental   # collect only
python -m autoresearch.cli --topic "..." --analyze                 # analyze only
python -m autoresearch.cli --topic "..." --package                 # package only
```

Output lands in `results/<topic-slug>-<timestamp>/` (same as
`python -m autoresearch.research`).

> If you prefer the manual step-by-step flow, continue with Step 0 below.

---

### Step 0 — Reset the workspace (required before every new topic)

> **Skip this and your new research will be contaminated by the previous run.**
> `papers/` accumulates old PDFs, `data/` retains stale artifacts, the prompt
> cache serves old LLM responses for similar-sounding prompts, and
> `templates/program.md` still describes the last topic.

```bash
python new_run.py --topic "Improving project-rules-generator skill for Claude"
```

What it does:
- Archives everything in `papers/` → `papers/.archive/<timestamp>/` (safe, not deleted)
- Clears stale `data/` artifacts (`all_docs*.txt`, `train.jsonl`, `val.jsonl`, etc.)
- Clears the LLM prompt cache (`dashboard/cache/prompts.jsonl`) so stale
  responses from a previous "skills" run don't bleed into the new one
- Rewrites the `## Topic` line in `templates/program.md`

Preview what it would do without making changes:
```bash
python new_run.py --topic "..." --dry-run
```

Retrying a failed run (keep cache to avoid redundant API calls):
```bash
python new_run.py --topic "..." --keep-cache
```

---

### Step 1 — Collect your sources

Collect from PDFs, web pages, and GitHub repos related to your topic.

**Incremental collect (recommended for re-runs):**

Use `python -m autoresearch.incremental_collect` when you're adding a few
new PDFs to an existing corpus — it skips sources whose content hash
matches the last run and only processes genuinely new or changed files:

```bash
python -m autoresearch.incremental_collect \
  --topic "Improving project-rules-generator skill for Claude" \
  --pdf-dir papers/ \
  --data-dir data/

# After adding a new PDF to papers/, only that file is processed.
# Re-running on the same unchanged directory: "Nothing new to collect."

# Force re-collect all sources (ignore the cache):
python -m autoresearch.incremental_collect --pdf-dir papers/ --force
```

Hash state is stored in `data/collect_metadata.jsonl` — delete it to
reset the incremental cache without affecting the corpus.

**Full collect (first run or when corpus needs rebuilding):**

```bash
# Option A: point at local PDFs you already have
python -m collector.ultimate_collector \
  --pdf-dir ~/Downloads/ \
  --output-dir data/

# Option B: scrape the web + specific GitHub repos
python -m collector.ultimate_collector \
  --queries "Claude SKILL.md best practices" \
             "project rules generator AGENTS.md" \
             "how to write cursorrules" \
  --github anthropics/anthropic-sdk-python \
           karpathy/autoresearch \
  --output-dir data/

# Option C: mix both
python -m collector.ultimate_collector \
  --pdf-dir papers/ \
  --queries "Claude skill description trigger" \
  --github Amitro123/project-rules-generator \
  --output-dir data/
```

The Claude skills PDF from your Downloads folder is a perfect source here:
```bash
# Already extracted — copy it into data/
cp ~/Downloads/The-Complete-Guide-to-Building-Skill-for-Claude.pdf papers/
python -m collector.ultimate_collector --pdf-dir papers/ --output-dir data/
```

> **Watch out for self-referential documents.** If you provide files like
> `CODE_REVIEW.md` or `UltimateDocResearcher_CR.md` as sources, their
> references to earlier topics ("Issue #18", "Skills") will leak into your
> corpus and contaminate the Q&A pairs. Only put actual research sources in
> `papers/`.

After collection, `data/all_docs.txt` contains everything separated by `<DOC_SEP>`.
The incremental collector appends to (rather than overwrites) this file, so
repeated partial runs accumulate content safely.

---

### Step 2 — Analyze / clean

```bash
python -c "
from collector.analyzer import analyze_corpus
import json
report = analyze_corpus('data/all_docs.txt', 'data/')
print(json.dumps(report, indent=2))
"
```

Output: `data/all_docs_cleaned.txt` — chunks that passed the quality filter.
Check `data/corpus_report.json` for per-document quality scores.
If a source scored < 0.25 it was filtered out — consider replacing it.

---

### Step 3 — Generate Q&A pairs

This is the step where you "ask the model what to research."
Three backends in priority order:

**Best quality — NotebookLM (free, needs browser auth once):**
```bash
python autoresearch/prepare.py \
  --corpus data/all_docs_cleaned.txt \
  --source-type notebooklm \
  --pdf-sources papers/The-Complete-Guide-to-Building-Skill-for-Claude.pdf \
  --max-pairs 100
```

> **What you gain from NotebookLM vs the other backends:**
>
> | | Heuristic | Local LLM | NotebookLM |
> |---|---|---|---|
> | How it reads your PDF | Regex on text fragments | Chunk-by-chunk (4K–8K tokens) | Whole document, all pages at once |
> | Question quality | Pattern-matched phrases | OK but misses cross-page context | Expert-level, spans full document |
> | Answers | Extracted text | LLM-generated | Explicitly marked correct options + hints |
> | Cost | Free | Free (Ollama) / paid (API) | **Free** (Google account) |
> | Setup | None | Ollama installed | One-time browser login |
>
> **Concrete impact on your results:** The Q&A pairs in `val.jsonl` are your
> research questions — they drive what the model learns and what code patterns
> get surfaced. NotebookLM questions are more specific, more meaningful, and
> cover the full depth of the source material. This produces noticeably higher
> `judge_avg_score` in `eval_report.json` and more actionable snippets in
> `code_suggestions.md`.
>
> **When to use it:** Any time you have PDFs (research papers, technical guides,
> documentation). NotebookLM is purpose-built for reading PDFs, so it
> outperforms any generic LLM on this task regardless of model size.
> If you only have scraped web content (no PDFs), fall back to `--source-type llm`.

**Good quality — local Ollama (free, no auth needed):**
```bash
python autoresearch/prepare.py \
  --corpus data/all_docs_cleaned.txt \
  --source-type llm \
  --model ollama:llama3.2 \
  --max-pairs 100
```

**Fastest — heuristic (no dependencies, lower quality):**
```bash
python autoresearch/prepare.py \
  --corpus data/all_docs_cleaned.txt \
  --max-pairs 100
```

Output: `data/train.jsonl` (90%) and `data/val.jsonl` (10%).

> **What "asking the model what to research" means:**
> The Q&A pairs in `val.jsonl` ARE your research questions.
> Open the file and read the questions — they surface the key concepts
> the corpus contains. These become your research agenda.
>
> For the project-rules-generator topic, the simulation produced:
> - "What aspects of a codebase are hardest for AI to infer automatically?"
> - "Rules files should be generated from the code, not from scratch — how?"
> - "What makes a rules file become stale?"
> - "What corpus sources are best for skills research?"
>
> These are the exact questions you need answered to improve your skill.

---

### Step 4 — Evaluate Q&A quality

```bash
# Free — local Ollama judge
python -m autoresearch.eval \
  --val-path data/val.jsonl \
  --judge-model ollama:llama3.2 \
  --max-samples 50 \
  --output-dir results/

# Cloud judge (better scoring accuracy)
python -m autoresearch.eval \
  --val-path data/val.jsonl \
  --judge-model claude-3-5-haiku-20241022 \
  --max-samples 50 \
  --output-dir results/
```

Check `results/eval_report.json`:
- `summary.avg_overall` — overall Q&A quality (1–5)
- `summary.worst_samples` — the weakest questions; these point to gaps in your corpus
- If pass_rate < 50%, your corpus is thin — go back to Step 1 and add more sources

---

### Step 5 — Generate code suggestions

```bash
python -m autoresearch.code_suggester \
  --corpus data/all_docs_cleaned.txt \
  --topic "Improving project-rules-generator skill for Claude" \
  --model ollama:llama3.2 \
  --n-suggestions 5 \
  --output results/code_suggestions.md
```

Open `results/code_suggestions.md` — you'll get 5 Python snippets
directly applicable to your project. For the skills creator topic,
expect suggestions like:

- A `scan_codebase()` function that reads files and extracts patterns
- A `generate_skill_description()` function that writes tight trigger descriptions
- A `test_skill_trigger()` function for evaluating if a description fires correctly
- A `build_rules_file()` function that produces CLAUDE.md from repo analysis

---

### Step 5b — Score output quality (standardized eval)

Once `code_suggestions.md` exists, score it against the 5-criteria spec.
This tells you whether the output is actually good enough to act on.

```bash
python -m eval.run_eval \
  --input results/code_suggestions.md \
  --judge ollama:llama3.2 \
  --threshold 3.5 \
  --output results/eval-report.json
```

The terminal prints a score table like:

```
──────────────────────────────────────────────────────
  📊  Eval Report  (✅ PASS)
──────────────────────────────────────────────────────
  Weighted score : 4.14 / 5.00

  clarity        [████░] 4/5  (×2.0)
  completeness   [████░] 4/5  (×1.5)
  actionability  [█████] 5/5  (×1.5)
  freshness      [███░░] 3/5  (×1.0)
  anti_patterns  [████░] 4/5  (×1.0)
──────────────────────────────────────────────────────
```

If **freshness** is low → add more recent sources to your corpus (2025/2026 docs).
If **anti_patterns** is low → re-run code_suggester with `--n-suggestions 7` and include an explicit anti-patterns prompt.
If **clarity** is low (most expensive criterion at ×2.0) → your corpus is probably too abstract; add more concrete how-to sources.

The full report is saved to `results/eval-report.json`.

---

### Step 5c — (Alternative) Generate a full research package

Instead of (or in addition to) Step 5, you can generate a complete set of
structured deliverables in one command. This is best when you want more than
just code snippets — for example, an implementation plan, a risk register,
or a benchmark comparison.

```bash
python -m autoresearch.research \
  --topic "Improving project-rules-generator skill for Claude"
```

The system **automatically classifies your topic** and generates only the
relevant files:

| Topic type | Example | Files generated |
|------------|---------|----------------|
| `code` | "Claude SDK tool use patterns" | SUMMARY + IMPLEMENTATION + NEXT_STEPS + CODE |
| `arch` | "Streaming data pipeline architecture" | SUMMARY + ARCHITECTURE + RISKS + NEXT_STEPS + CODE |
| `process` | "Fine-tuning LLMs with RLHF" | SUMMARY + IMPLEMENTATION + RISKS + NEXT_STEPS + CODE |
| `market` | "Survey of LLM eval frameworks" | SUMMARY + BENCHMARKS + RISKS + NEXT_STEPS + CODE |

Output goes to `results/<topic-slug>-<timestamp>/`. Open the **Research
Packages** tab in the dashboard to browse and download any deliverable.

```bash
# Check what type your topic would classify as before running:
python -c "
from research_deliverables.classify_topic import classify_topic
ds = classify_topic('your topic here')
print(ds.research_type, ds.deliverables)
"

# Full pipeline in one command (collect + analyze + generate):
python -m autoresearch.research \
  --topic "multi-tenant RAG with Claude tool use" \
  --collect \
  --pdf-dir papers/ \
  --model gemini-2.5-flash-lite

# Skip code suggestions (faster, useful if you ran code_suggester separately):
python -m autoresearch.research --topic "RAG architecture" --no-code
```

You can also eval each deliverable individually:
```bash
python -m eval.run_eval --input results/<run-id>/IMPLEMENTATION.md --threshold 3.5
python -m eval.run_eval --input results/<run-id>/RISKS.md --threshold 3.5
```

---

### Step 6 — (Optional) Train a fine-tuned model

Only needed if you want a model that answers questions about your corpus
directly, rather than just code suggestions.

**Local (CPU, smoke test):**
```bash
python autoresearch/train.py --topic "skills creator" --iterations 1
```

**Remote free GPU — Kaggle:**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
python api-triggers/trigger_kaggle.py \
  --topic "skills creator improvement" \
  --iterations 5 \
  --download-results
```

**Remote GPU — Modal (recommended over Kaggle):**
```bash
pip install modal
modal setup                    # one-time auth
# then run research_loop with Modal backend (see AGENTS.md Phase 6)
```

---

## Applying results to `project-rules-generator`

After running the simulation, the key research findings were:

**1. Reconnaissance before generation**
The rules generator should read the actual repo before writing rules.
Add a "scan phase" to your skill:
```python
# In your SKILL.md instructions, before generating rules:
# 1. List top-level directories
# 2. Read 3-5 representative source files
# 3. Read existing tests to understand testing patterns
# 4. Check package.json / pyproject.toml / requirements.txt for stack
# THEN generate rules based on what you found
```

**2. Description is the most important field**
Your skill's trigger description determines everything.
Current weak pattern: "Generates project rules"
Better: "Generates CLAUDE.md, AGENTS.md, and .cursorrules files for a codebase.
Use when asked to create project rules, set up AI context files,
generate coding guidelines, or document project conventions for AI tools."

**3. Anti-patterns section**
The highest-value addition to any rules file is what NOT to do.
Add to your generator prompt: "After identifying patterns, explicitly list
3 anti-patterns observed in the codebase and add them to the rules file."

**4. Freshness metadata**
Add a generated timestamp and file count to the rules file header.
This makes stale rules visible:
```markdown
<!-- Generated: 2026-03-14 | Files scanned: 47 | Re-run when structure changes -->
```

**5. Self-test loop**
Use the eval pipeline to test your own skill:
```bash
# Generate rules for a test repo
# Feed rules + a task prompt to Claude
# Score output with LLM judge
# Iterate on SKILL.md until score > 4.0
python -m autoresearch.eval \
  --val-path data/skills_test_cases.jsonl \
  --judge-model ollama:llama3.2
```

---

## Quick Reference

| Goal | Command |
|------|---------|
| **Phase 13 — full pipeline (one command)** | `python -m autoresearch.cli --topic "X" --full --incremental` |
| Phase 13 — deliverables only | `python -m autoresearch.cli --topic "X"` |
| Phase 13 — collect step only | `python -m autoresearch.cli --topic "X" --collect --incremental` |
| Phase 13 — force re-collect | `python -m autoresearch.cli --topic "X" --full --incremental --force-recollect` |
| **Incremental collect (standalone)** | `python -m autoresearch.incremental_collect --pdf-dir papers/ --data-dir data/` |
| Incremental collect (force) | `python -m autoresearch.incremental_collect --pdf-dir papers/ --force` |
| **Reset before new topic** | `python new_run.py --topic "your topic"` |
| Reset (keep cache) | `python new_run.py --topic "your topic" --keep-cache` |
| Reset dry-run | `python new_run.py --topic "your topic" --dry-run` |
| Collect PDFs only | `python -m collector.ultimate_collector --pdf-dir papers/ --output-dir data/` |
| Collect from web | `python -m collector.ultimate_collector --queries "topic" --output-dir data/` |
| Clean corpus | `python -c "from collector.analyzer import analyze_corpus; analyze_corpus('data/all_docs.txt', 'data/')"` |
| Prepare (heuristic) | `python autoresearch/prepare.py --corpus data/all_docs_cleaned.txt` |
| Prepare (Ollama) | `python autoresearch/prepare.py --source-type llm --model ollama:llama3.2` |
| Prepare (NotebookLM) | `python autoresearch/prepare.py --source-type notebooklm --pdf-sources papers/*.pdf` |
| Eval Q&A quality | `python -m autoresearch.eval --val-path data/val.jsonl --judge-model ollama:llama3.2` |
| Code suggestions | `python -m autoresearch.code_suggester --corpus data/all_docs_cleaned.txt --model ollama:llama3.2` |
| **Multi-format package** | `python -m autoresearch.research --topic "your topic"` |
| Multi-format (full pipeline) | `python -m autoresearch.research --topic "X" --collect --pdf-dir papers/` |
| Multi-format (no code) | `python -m autoresearch.research --topic "X" --no-code` |
| Check topic type | `python -c "from research_deliverables.classify_topic import classify_topic; print(classify_topic('topic').research_type)"` |
| **Score output quality** | `python -m eval.run_eval --input results/code_suggestions.md --judge ollama:llama3.2` |
| Score any deliverable | `python -m eval.run_eval --input results/<run-id>/RISKS.md --judge ollama:llama3.2` |
| **Batch score outputs** | `python -m eval.run_eval --input results/*.md --judge ollama:llama3.2` |
| Check LLM setup | `python -m autoresearch.llm_client check` |
| Auto-detect model | `python -m autoresearch.llm_client auto` |
| Full loop (1 iter) | `python autoresearch/train.py --topic "X" --iterations 1 --judge-model ollama:llama3.2` |
| **Launch dashboard** | `streamlit run dashboard/app.py` |
| **Seed demo data** | `python dashboard/seed_demo.py` |
| **Skip similar topic** | `python autoresearch/train.py --topic "X" --skip-if-similar` |
| **Cache stats** | `python -c "from memory.cache import PromptCache; print(PromptCache().stats())"` |

---

## Dashboard & Memory System

The dashboard gives you a visual history of every research run, lets you
compare metrics across topics, and warns you when you're about to re-run
research you've already done.

### Launch

```bash
# First time: seed realistic demo data so the dashboard isn't empty
python dashboard/seed_demo.py

# Launch (requires streamlit)
pip install streamlit pandas
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### Pages

**Recent Runs** — table of all runs with topic, status, score, and duration.
Click the download button next to any row to grab its output file directly.

**Run Explorer** — filter by topic or status, view per-iteration metrics as
a table and a line chart. Useful for spotting which iteration produced the
best result.

**Metrics** — aggregate charts: avg score over time, pass-rate trend, and
a per-topic bar chart. Helps you see whether your corpus or prompts are
improving across runs.

**New Run** — enter a topic and click "Check for similar runs" before
starting. If a past run covered similar ground (cosine similarity ≥ 0.85),
the dashboard warns you and offers to reuse the existing output instead of
burning API credits. The **Incremental collect** checkbox (default: on)
makes the generated command use `autoresearch.cli --incremental`, so only
new or changed sources are fetched. Uncheck it for a full re-collect, or
check **Force re-collect** to ignore the incremental cache entirely.

**Research Packages** — browse all multi-format packages generated by
`python -m autoresearch.research`. Select a package from the dropdown to
view its deliverables (SUMMARY, ARCHITECTURE, IMPLEMENTATION, RISKS,
BENCHMARKS, NEXT_STEPS, CODE) in tabs, with per-file download buttons and
a regenerate command.

### How the memory system works

Every `research_loop()` call automatically integrates with the memory
system (gracefully skipped if the `memory/` module is unavailable):

1. Checks `dashboard/runs.db` for past runs with a similar topic
2. If `--skip-if-similar` is set and a match is found, exits early and
   prints the previous result path
3. Registers the new run with a unique ID
4. Logs per-iteration metrics (score, tokens, latency) as the loop runs
5. Marks the run complete (or failed) when it finishes

```bash
# Memory-aware run: skip if similar topic was researched within 7 days
python autoresearch/train.py \
  --topic "Claude skill description best practices" \
  --iterations 3 \
  --judge-model ollama:llama3.2 \
  --similarity-threshold 0.85 \
  --skip-if-similar
```

### Prompt cache

All `chat()` calls in `autoresearch/llm_client.py` are automatically
cached in `memory/prompts.db` (SQLite). The cache does two lookups
before hitting the LLM:

1. **Exact** — SHA1 hash of the full prompt + model string → O(1) lookup
2. **Fuzzy** — cosine similarity ≥ 0.92 against cached prompts for
   the same model (catches rephrased-but-identical questions)

> **Note:** The cache migrated from `dashboard/cache/prompts.jsonl`
> (O(n) rewrite on every write) to SQLite in `memory/prompts.db`
> (O(1) upserts). If you have an old `prompts.jsonl` it is auto-migrated
> on first run — you can delete it afterwards.

Cache hits are returned instantly with zero API cost. To bypass the cache
for a single call, pass `use_cache=False`:

```python
from autoresearch.llm_client import chat

# bypass cache (e.g. when you want a fresh answer)
reply = chat(messages, model="ollama:llama3.2", use_cache=False)
```

> **Important — new topic = clear the cache.**
> The fuzzy matcher at 0.92 similarity will serve cached responses from a
> previous "skills" run when you start a new "skills" run, even if the corpus
> is completely different. `new_run.py` clears the cache automatically.
> If you need to keep the cache (e.g. retrying a failed run), use:
> ```bash
> python new_run.py --topic "..." --keep-cache
> ```

To inspect or clear the cache manually:

```python
from memory.cache import PromptCache
c = PromptCache()
print(c.stats())   # {"total_entries": 42, "total_hits": 18, ...}
c.clear()          # wipe all cached responses
```

---

## Decision Tree: Which LLM backend to use?

```
Do you have Ollama running locally?
  YES → use ollama:llama3.2 for everything (free, fast, private)
  NO  →
    Do you have ANTHROPIC_API_KEY?
      YES → use claude-3-5-haiku-20241022 (best quality, ~$0.01 per run)
      NO  →
        Do you have GOOGLE_API_KEY?
          YES → use gemini-1.5-flash (free tier available; expect 429 rate-limit
                retries on free keys — built-in backoff handles this automatically)
          NO  →
            Do you have OPENAI_API_KEY?
              YES → use gpt-4o-mini
              NO  → use heuristic (always works, lower Q&A quality)

For prepare.py specifically:
  Do you have PDFs and notebooklm-py installed?
    YES → --source-type notebooklm (highest Q&A quality, free)
    NO  → --source-type llm --model ollama:llama3.2
```

---

## Understanding the output files

| File | What it contains | When to look at it |
|------|-----------------|-------------------|
| `data/collect_metadata.jsonl` | Incremental collect hash cache (Phase 13) | Delete to reset incremental state |
| `data/all_docs.txt` | Raw collected documents | Debug collection issues |
| `data/all_docs_cleaned.txt` | Quality-filtered chunks | Input to prepare.py |
| `data/external_docs.txt` | External-only chunks (tagged by analyzer) | Used for 70% of LLM window |
| `data/corpus_report.json` | Per-doc quality scores + external fraction | Find weak/internal-only sources |
| `data/train.jsonl` | Training Q&A pairs | Input to train.py |
| `data/val.jsonl` | Your research questions | READ THIS — it's your agenda |
| `results/eval_report.json` | Q&A judge scores per sample | Find corpus gaps |
| `results/eval_report.json` → `worst_samples` | Lowest-scored Q&As | Where to add more sources |
| `results/code_suggestions.md` | Copy-paste code snippets | Quick deliverable |
| `results/eval-report.json` | 6-criteria output quality score | Is the output good enough? |
| `results/eval-report.json` → `criterion_scores` | Per-criterion breakdown | What specifically to improve |
| `results/<run-id>/SUMMARY.md` | Executive overview + key findings | Start here when sharing results |
| `results/<run-id>/ARCHITECTURE.md` | Component diagram, data flow, trade-offs | Architecture / system design topics |
| `results/<run-id>/IMPLEMENTATION.md` | Step-by-step plan, code patterns, pitfalls | Implementation / process topics |
| `results/<run-id>/RISKS.md` | Risk register table + mitigations | Before starting any major build |
| `results/<run-id>/BENCHMARKS.md` | Comparison tables + performance numbers | Survey / market topics |
| `results/<run-id>/NEXT_STEPS.md` | Prioritised actions (this week / 1 month) | After reading the summary |
| `results/<run-id>/CODE/code_suggestions.md` | Copy-paste snippets inside a package | Your main code deliverable |
| `results/<run-id>/metadata.json` | Run ID, topic type, corpus stats, errors | Debug or re-run a package |
| `results/results.tsv` | Training metrics per iteration | Track model improvement |
| `dashboard/runs.db` | SQLite history of all runs + metrics | Query with any SQLite tool |
| `memory/prompts.db` | SQLite prompt cache (O(1) lookups, fuzzy match) | Inspect or clear cache |

---

## Troubleshooting

**"Judge call failed: Tunnel connection failed"**
→ Your network blocks outbound HTTP. Use Ollama (local, no network needed):
`--judge-model ollama:llama3.2`

**"notebooklm-py not installed"**
→ `pip install "notebooklm-py[browser]" && playwright install chromium && notebooklm login`

**"Corpus not found"**
→ Run the analyzer step first: `analyze_corpus('data/all_docs.txt', 'data/')`

**"Generated 0 Q&A pairs"**
→ Corpus chunks are too short (<150 chars). Lower the threshold in analyzer.py or add more content.

**Code suggestions are generic skeletons**
→ No LLM available. Run `python -m autoresearch.llm_client check` to diagnose,
then install Ollama or set an API key.

**NotebookLM returned 0 pairs**
→ Unofficial API changed. Fall back to `--source-type llm`. Check notebooklm-py
GitHub for updates.

**`eval/run_eval.py` scores everything 2-3 with heuristics**
→ No LLM judge available. Run `python -m autoresearch.llm_client check` to see
what's reachable, then pass `--judge ollama:llama3.2` (local) or set an API key.

**Output-quality eval fails threshold but Q&A eval passed**
→ These measure different things. Q&A eval scores training data quality.
Output eval scores whether `code_suggestions.md` is clear, complete, and actionable.
Add higher-quality corpus sources, then re-run code_suggester.

**Results look like a previous run / wrong topic appearing in output**
→ You skipped `new_run.py`. Old PDFs in `papers/`, stale `data/` files, or a
cached prompt response from the last run are bleeding into this one.
Fix: `python new_run.py --topic "your topic"` then start from Step 1.

**Gemini 429 "quota_id: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"**
→ Free-tier keys are rate-limited per minute. The client retries automatically
with exponential back-off (15 s → 30 s → 60 s → 120 s) — you'll see
`⚠️ Gemini rate limit` in stderr. If it keeps failing after all retries,
wait 2–3 minutes before re-running. To permanently avoid this, enable
Pay-as-you-go billing on the project in Google Cloud Console.

**Research package generates wrong deliverable type**
→ `classify_topic()` uses keyword matching. Check what it returns:
`python -c "from research_deliverables.classify_topic import classify_topic; print(classify_topic('your topic').research_type)"`
If the classification is wrong, override with a more specific topic string
(e.g. "Architecture of …" for arch, "Survey of …" for market).

**Research package deliverable is shallow / LLM-only content**
→ The generators are only as good as the corpus. If ARCHITECTURE.md reads
like generic advice, your corpus has too few external architecture sources.
Go back to Step 1, add PDFs or GitHub repos on that specific aspect, then
re-run: `python -m autoresearch.research --topic "..." --no-code`

**`jinja2` not found when running `autoresearch.research`**
→ `pip install jinja2` (or `pip install -r requirements.txt`).
The generators fall back to a plain key=value format if Jinja2 is missing,
but the output won't be properly formatted Markdown.

**"Research Packages" dashboard tab shows no packages**
→ No `metadata.json` files found under `results/`. Generate a package first:
`python -m autoresearch.research --topic "your topic"`

**Incremental collect says "Nothing new to collect" but corpus is stale**
→ The hash cache in `data/collect_metadata.jsonl` thinks all sources are
unchanged. Delete that file to reset it (the corpus `all_docs.txt` is
untouched), then re-run — or use `--force` / `--force-recollect` to
ignore the cache for that run only.

**`python -m autoresearch.cli` exits immediately without generating files**
→ Without `--full` or individual step flags it defaults to `--package`
(deliverables only). If no corpus exists yet, add `--collect` or `--full`.
Run with `--help` to see all options.
