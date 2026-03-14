# UltimateDocResearcher — End-to-End Guide

> How to go from "I want to improve X" to working code suggestions,
> using your own project as the research subject.

---

## The Big Picture

```
Your question / project
        │
        ▼
  1. COLLECT       ← PDFs, URLs, GitHub repos, your own codebase
        │
        ▼
  2. ANALYZE       ← quality filter, dedup, chunking
        │
        ▼
  3. PREPARE       ← Q&A generation  (NotebookLM → LLM → heuristic)
        │
        ▼
  4. EVAL (Q&A)    ← LLM-as-Judge scores your Q&A pair quality
        │
        ▼
  5. SUGGEST       ← code snippets derived from your research
        │                        ↑ prompt cache (dashboard/cache/)
        ▼
  6. EVAL (output) ← 5-criteria score on code_suggestions.md
        │
        ▼
  results/eval-report.json  +  results/code_suggestions.md
        │
        ▼
  7. DASHBOARD     ← run history, metrics, topic dedup (dashboard/app.py)
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

# Option D — NotebookLM (best for PDFs, needs browser auth once)
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

### Step 1 — Collect your sources

Collect from PDFs, web pages, and GitHub repos related to your topic.

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

After collection, `data/all_docs.txt` contains everything separated by `<DOC_SEP>`.

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
NotebookLM reads your PDFs deeply and generates quiz-quality questions —
far better than anything a small model can produce.

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
| Collect PDFs only | `python -m collector.ultimate_collector --pdf-dir papers/ --output-dir data/` |
| Collect from web | `python -m collector.ultimate_collector --queries "topic" --output-dir data/` |
| Clean corpus | `python -c "from collector.analyzer import analyze_corpus; analyze_corpus('data/all_docs.txt', 'data/')"` |
| Prepare (heuristic) | `python autoresearch/prepare.py --corpus data/all_docs_cleaned.txt` |
| Prepare (Ollama) | `python autoresearch/prepare.py --source-type llm --model ollama:llama3.2` |
| Prepare (NotebookLM) | `python autoresearch/prepare.py --source-type notebooklm --pdf-sources papers/*.pdf` |
| Eval Q&A quality | `python -m autoresearch.eval --val-path data/val.jsonl --judge-model ollama:llama3.2` |
| Code suggestions | `python -m autoresearch.code_suggester --corpus data/all_docs_cleaned.txt --model ollama:llama3.2` |
| **Score output quality** | `python -m eval.run_eval --input results/code_suggestions.md --judge ollama:llama3.2` |
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
burning API credits.

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
cached in `dashboard/cache/prompts.jsonl`. The cache does two lookups
before hitting the LLM:

1. **Exact** — SHA1 hash of the full prompt + model string
2. **Fuzzy** — cosine similarity ≥ 0.92 against all cached prompts for
   the same model (catches rephrased-but-identical questions)

Cache hits are returned instantly with zero API cost. To bypass the cache
for a single call, pass `use_cache=False`:

```python
from autoresearch.llm_client import chat

# bypass cache (e.g. when you want a fresh answer)
reply = chat(messages, model="ollama:llama3.2", use_cache=False)
```

To inspect or clear the cache:

```python
from memory.cache import PromptCache
c = PromptCache()
print(c.stats())   # {"total_entries": 42, "total_size_bytes": 18400, ...}
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
| `data/all_docs.txt` | Raw collected documents | Debug collection issues |
| `data/all_docs_cleaned.txt` | Quality-filtered chunks | Input to prepare.py |
| `data/corpus_report.json` | Per-doc quality scores | Find weak sources |
| `data/train.jsonl` | Training Q&A pairs | Input to train.py |
| `data/val.jsonl` | Your research questions | READ THIS — it's your agenda |
| `results/eval_report.json` | Q&A judge scores per sample | Find corpus gaps |
| `results/eval_report.json` → `worst_samples` | Lowest-scored Q&As | Where to add more sources |
| `results/code_suggestions.md` | Copy-paste code snippets | Your main deliverable |
| `results/eval-report.json` | 5-criteria output quality score | Is the output good enough? |
| `results/eval-report.json` → `criterion_scores` | Per-criterion breakdown | What specifically to improve |
| `results/results.tsv` | Training metrics per iteration | Track model improvement |
| `dashboard/runs.db` | SQLite history of all runs + metrics | Query with any SQLite tool |
| `dashboard/cache/prompts.jsonl` | Cached LLM prompt→response pairs | Inspect cache hits/misses |

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
