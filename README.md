## Architecture

```mermaid
graph TD
    subgraph Collector
        C[UltimateCollector] --> P[PDF/Local]
        C --> W[Web Scraper]
        C --> D[Google Drive]
        C --> G[GitHub Repos]
    end

    C --> |data/all_docs.txt| A[Autoresearch Loop]
    
    subgraph Research
        A --> T[Train / LoRA]
        A --> E[Evaluate]
        A --> Q[Query Generation]
    end
    
    Q --> |knowledge gaps| C
    T --> |Results| R[results/metrics.tsv]
```

## 🚀 Quick Start

### 1. Local Setup
```bash
# Clone & install
git clone https://github.com/Amitro123/UltimateDocResearcher
cd ultimate-doc-researcher
pip install -r requirements.txt

# Run a mock research cycle
python collector/run_mock.py --topic "AI engineering"
```

### 2. Collect documents

```bash
# Scrape web + reddit + GitHub on a topic
python -m collector.ultimate_collector \
  --queries "Claude prompt engineering" "LoRA fine-tuning best practices" \
  --reddit MachineLearning LocalLLaMA \
  --github karpathy/autoresearch \
  --output-dir data/

# Or include local PDFs
python -m collector.ultimate_collector \
  --pdf-dir papers/ \
  --queries "transformer architecture" \
  --output-dir data/
```

### 3. Prepare training data

```bash
python autoresearch/prepare.py \
  --corpus data/all_docs_cleaned.txt \
  --output-dir data/ \
  --max-pairs 500
```

### 4a. Choose your LLM (OpenAI / Anthropic / Ollama)

All modules that call an LLM (`eval.py`, `code_suggester.py`, `prepare.py`) share the same model-string convention:

| Model string | Provider | Cost |
|---|---|---|
| `gpt-4o-mini` | OpenAI (`OPENAI_API_KEY`) | ~$0.15/1M tokens |
| `claude-3-5-haiku-20241022` | Anthropic (`ANTHROPIC_API_KEY`) | ~$0.25/1M tokens |
| `ollama:llama3.2` | Local Ollama | **Free** |
| `ollama:mistral@http://host:11434` | Remote Ollama | **Free** |

**Setting up Ollama (free, runs locally, no API key needed):**

```bash
# 1. Install Ollama
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh
# Windows: https://ollama.com/download

# 2. Pull a model (one-time download, ~2GB for llama3.2)
ollama pull llama3.2       # recommended — fast + good quality
ollama pull mistral        # alternative
ollama pull phi4           # smallest / fastest

# 3. Verify it's reachable
python -m autoresearch.llm_client check

# 4. Test a quick prompt
python -m autoresearch.llm_client chat \
  --model ollama:llama3.2 \
  --prompt "Explain LoRA fine-tuning in one sentence."

# 5. Auto-detect best available model (Ollama → Anthropic → OpenAI)
python -m autoresearch.llm_client auto
```

### 4b. Evaluate with LLM-as-a-Judge

After preparing data (or after training), run the judge to score your val set:

```bash
# Free — local Ollama
python -m autoresearch.eval \
  --val-path data/val.jsonl \
  --judge-model ollama:llama3.2 \
  --max-samples 50

# OpenAI
python -m autoresearch.eval \
  --val-path data/val.jsonl \
  --judge-model gpt-4o-mini \
  --max-samples 50

# Anthropic Claude
python -m autoresearch.eval \
  --val-path data/val.jsonl \
  --judge-model claude-3-5-haiku-20241022 \
  --max-samples 50

# Point at a trained model for end-to-end eval
python -m autoresearch.eval \
  --val-path data/val.jsonl \
  --model-path models/lora_adapter \
  --judge-model ollama:llama3.2
```

Output: `results/eval_report.json` with per-sample scores + summary.

### 4c. Generate code suggestions from research corpus

After collecting and cleaning your corpus, generate ready-to-use Python snippets:

```bash
# Free — local Ollama
python -m autoresearch.code_suggester \
  --corpus data/all_docs_cleaned.txt \
  --model ollama:llama3.2 \
  --n-suggestions 5

# Or with a cloud provider
python -m autoresearch.code_suggester \
  --corpus data/all_docs_cleaned.txt \
  --model claude-3-5-haiku-20241022 \
  --n-suggestions 5
```

Output: `results/code_suggestions.md` — copy-paste Python code examples derived from your research topic. If your corpus covers Claude tool use, you get annotated SDK snippets. LoRA fine-tuning corpus → training loop examples. Etc.

### 4d. Score output quality with the standardized eval framework

After generating code suggestions (or any research output), score it against
the 5-criteria spec:

```bash
python -m eval.run_eval \
  --input results/code_suggestions.md \
  --judge ollama:llama3.2 \
  --threshold 3.5 \
  --output results/eval-report.json
```

Exit code `0` = pass, `1` = fail — safe to use in CI pipelines.

### 5. Train on Kaggle (remote, no local GPU)

```bash
# Set required secrets
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
export GITHUB_TOKEN=ghp_...

python api-triggers/trigger_kaggle.py \
  --topic "Claude skills optimization" \
  --iterations 20 \
  --github-repo yourusername/ultimate-doc-researcher \
  --download-results
```

### 6. Or trigger via GitHub Actions

```bash
gh workflow run research.yml \
  -f topic="Claude skills optimization" \
  -f iterations=20
```

Then watch: **Actions → UltimateDocResearcher → Run #N**

---

## Dashboard & Memory System

### Launch the dashboard

```bash
pip install streamlit pandas
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` with four views:

- **Recent Runs** — table of all runs with score/status, download links for `code_suggestions.md` and `eval-report.json`
- **Run Explorer** — drill into any run's per-iteration metrics with a val_score line chart
- **Metrics** — avg_score and pass_rate over time, score by topic bar chart
- **New Run** — topic input, similarity check against past runs, one-click launch

### Seed demo data (first-time setup)

```bash
python dashboard/seed_demo.py    # adds 8 realistic demo runs to runs.db
```

### How the memory system works

Every `research_loop()` call automatically:
1. Checks `dashboard/runs.db` for past runs with a similar topic (cosine similarity on TF-IDF)
2. Registers a new run row (`status=running`) at the start
3. Logs per-iteration metrics as they complete
4. Marks the run `completed` with final scores when done

```bash
# Skip if a similar run already exists (≥80% similarity)
python autoresearch/train.py \
  --topic "Building Claude skills" \
  --iterations 3 \
  --skip-if-similar

# Lower the threshold for stricter deduplication
python autoresearch/train.py \
  --topic "Claude SKILL.md optimization" \
  --similarity-threshold 0.6
```

### Prompt cache

The `PromptCache` stores LLM prompt→response pairs in `dashboard/cache/prompts.jsonl`, avoiding redundant API calls:

```python
from memory.cache import PromptCache

cache = PromptCache()
hit = cache.get_fuzzy("Explain LoRA fine-tuning", threshold=0.85)
if hit:
    response = hit["response"]
else:
    response = call_llm(...)
    cache.set("Explain LoRA fine-tuning", response, model="ollama:llama3.2")
```

---

## Project Structure

```
ultimate-doc-researcher/
├── collector/
│   ├── ultimate_collector.py   # Main orchestrator
│   ├── scraper.py              # Async web/reddit/github scraper
│   ├── drive_extractor.py      # Google Drive + Colab/Kaggle mounts
│   └── analyzer.py             # Quality filter, chunking, dedup
├── autoresearch/
│   ├── prepare.py              # Q&A generation from corpus
│   ├── train.py                # LoRA training loop + results.tsv
│   ├── eval.py                 # LLM-as-a-Judge evaluator
│   ├── code_suggester.py       # Post-research code suggestion engine
│   └── llm_client.py           # Unified LLM router (OpenAI/Anthropic/Ollama)
├── eval/
│   ├── eval_spec.yaml          # 5-criteria evaluation spec with weights
│   ├── run_eval.py             # Standardized eval runner CLI
│   └── test_cases/             # Sample outputs for manual/CI testing
├── memory/
│   ├── memory.py               # SQLite run history + topic similarity search
│   └── cache.py                # Prompt cache (exact + fuzzy matching)
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   ├── seed_demo.py            # Populate runs.db with demo data
│   ├── runs.db                 # SQLite run history (auto-created)
│   └── cache/
│       └── prompts.jsonl       # Cached LLM prompt→response pairs
├── templates/
│   ├── program.md              # Active research program
│   └── program_templates.py    # 4 built-in programs + generator
├── api-triggers/
│   ├── trigger_kaggle.py       # Push/poll/download Kaggle kernels
│   └── poll_results.py         # Results polling + git sync
├── .github/
│   └── workflows/
│       └── research.yml        # GitHub Actions (dispatch + cron)
├── results/
│   └── results.tsv             # val_score per iteration
├── demo/
│   └── demo.ipynb              # End-to-end walkthrough
├── AGENTS.md                   # Phase plans + architecture notes
├── Dockerfile                  # Local dev container
└── requirements.txt
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `KAGGLE_USERNAME` | For remote training | Kaggle username |
| `KAGGLE_API_TOKEN` | For remote training | Kaggle API token |
| `GITHUB_TOKEN` | For result commits / higher rate limits | GitHub PAT |
| `OPENAI_API_KEY` | Optional | Q&A generation, LLM judge, code suggestions |
| `ANTHROPIC_API_KEY` | Optional | Claude as judge / code suggester |
| `GOOGLE_API_KEY` | Optional | Google Custom Search |
| `GOOGLE_CX` | Optional | Google CSE engine ID |
| `GDRIVE_SA_KEY_PATH` | Optional | Service account JSON for Drive |

### Research Programs

```bash
# List available programs
python templates/program_templates.py --list
# claude-skills-optimizer
# mcp-agent-orchestration
# openclaw-production
# local-llm-fine-tuning

# Switch active program
python templates/program_templates.py \
  --program mcp-agent-orchestration \
  --output templates/program.md
```

---

## Results Format

`results/results.tsv` — tab-separated, one row per training iteration:

| Column | Description |
|--------|-------------|
| `iteration` | Loop counter (1..N) |
| `train_loss` | Training cross-entropy loss |
| `val_loss` | Validation loss |
| `val_score` | Normalised score 0–1 (higher = better) |
| `train_samples` | Number of training Q&A pairs |
| `elapsed_seconds` | Wall-clock training time |
| `topic` | Research topic |
| `timestamp` | ISO UTC timestamp |
| `judge_pass_rate` | Fraction of val samples passing judge threshold (eval.py) |
| `judge_avg_score` | Mean overall judge score 1–5 (eval.py) |

`results/eval_report.json` — per-sample Q&A judge output (from `autoresearch/eval.py`, runs inside the training loop):

| Field | Description |
|-------|-------------|
| `summary.avg_overall` | Mean judge score across all val samples |
| `summary.avg_accuracy` | Mean accuracy score |
| `summary.avg_relevance` | Mean relevance score |
| `summary.avg_completeness` | Mean completeness score |
| `summary.pass_rate` | Fraction of samples ≥ threshold |
| `summary.worst_samples` | 3 lowest-scoring questions (corpus gap signals) |
| `samples[]` | Per-sample question, reference, model answer, scores |

`results/code_suggestions.md` — Markdown file with N Python code snippets derived from the research corpus.

`results/eval-report.json` — standardized 5-criteria output quality report (from `eval/run_eval.py`, run separately on any output file):

| Field | Description |
|-------|-------------|
| `summary.weighted_avg` | Weighted average score across 5 criteria |
| `summary.passed` | bool: weighted_avg ≥ threshold |
| `summary.criterion_scores` | Per-criterion scores (1–5) |
| `criteria[].reasoning` | Judge's one-sentence rationale per criterion |

---

## Standardized Eval Framework

Score any research output against 5 fixed criteria defined in `eval/eval_spec.yaml`:

| Criterion | Weight | What it checks |
|-----------|--------|---------------|
| Clarity | 2.0 | No ambiguity — rules are crystal clear |
| Completeness | 1.5 | 90%+ of corpus patterns covered |
| Actionability | 1.5 | Copy-paste ready, no boilerplate |
| Freshness | 1.0 | 2026 patterns, no deprecated APIs |
| Anti-patterns | 1.0 | Explicitly warns against common mistakes |

```bash
# Score any output (uses heuristic fallback if no LLM)
python -m eval.run_eval \
  --input results/code_suggestions.md \
  --judge ollama:llama3.2 \
  --threshold 3.5 \
  --output results/eval-report.json

# With Anthropic judge
python -m eval.run_eval \
  --input results/code_suggestions.md \
  --judge claude-3-5-haiku-20241022

# Batch — score multiple files
python -m eval.run_eval --input results/*.md --judge ollama:llama3.2
```

Exit code is `0` if all files pass, `1` if any fail — CI-friendly.

---

## Docker (local dev)

```bash
docker build -t ultimate-doc-researcher .

docker run --rm \
  -v $(pwd)/data:/app/data \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  ultimate-doc-researcher \
  --queries "Claude skills" --reddit MachineLearning
```

---

## Roadmap

- [x] Phase 1: UltimateCollector (PDF/web/Drive/GitHub)
- [x] Phase 2: Remote Kaggle execution + GitHub Actions
- [x] Phase 3: Research program templates
- [x] Phase 5: LLM-as-a-Judge eval (`eval.py`)
- [x] Phase 5: Post-research code suggestions (`code_suggester.py`)
- [x] Phase 5: Unified LLM client (Ollama / Anthropic / OpenAI)
- [x] Phase 5: NotebookLM Q&A backend for `prepare.py`
- [x] Phase 5: Standardized 5-criteria eval framework (`eval/`)
- [x] Phase 6: Streamlit dashboard (`dashboard/app.py`)
- [x] Phase 6: SQLite run history + topic similarity (`memory/memory.py`)
- [x] Phase 6: Prompt cache with fuzzy matching (`memory/cache.py`)
- [ ] Phase 7: End-to-end CI tests
- [ ] Phase 6: Iterative corpus expansion (gap-driven re-collection)
- [ ] Phase 6: Multi-model ensemble with mergekit
- [ ] Phase 6: Streaming results dashboard

See [AGENTS.md](AGENTS.md) for detailed phase plans.

---

## License

MIT — fork freely, build cool stuff.
