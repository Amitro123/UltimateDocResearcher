# AGENTS.md — UltimateDocResearcher Phase Plans

> This file documents the autonomous agent architecture and phase-by-phase
> implementation plan. Updated after each phase is shipped.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    UltimateDocResearcher                     │
├────────────────┬────────────────┬────────────────────────────┤
│  collector/    │  autoresearch/ │  api-triggers/             │
│                │                │                            │
│  UltimateCol-  │  prepare.py    │  trigger_kaggle.py         │
│  lector        │  ─ Q&A gen    │  ─ push kernel            │
│  ├─ PDF        │  ─ train/val  │  ─ poll status            │
│  ├─ Web        │               │  ─ download results       │
│  ├─ Drive      │  train.py      │                            │
│  └─ GitHub     │  ─ unsloth   │  .github/workflows/        │
│                │  ─ LoRA      │  ─ workflow_dispatch       │
│  analyzer.py   │  ─ loop      │  ─ scheduled cron         │
│  ─ quality    │  ─ git commit │                            │
│  ─ chunking   │               │                            │
└────────────────┴────────────────┴────────────────────────────┘
                        │
                   results/
                   ─ results.tsv (iteration × metrics)
                   ─ models/     (LoRA adapters)
```

---

## ✅ Phase 1 — UltimateCollector (SHIPPED)

**Goal:** Collect documents from all sources into `data/all_docs.txt`.

### What was built
- `collector/ultimate_collector.py` — orchestrator class
- `collector/scraper.py` — async web/reddit/github scraper
- `collector/drive_extractor.py` — Google Drive + Colab/Kaggle mounts
- `collector/analyzer.py` — quality scoring, dedup, chunking

### Usage
```bash
python -m collector.ultimate_collector \
  --pdf-dir papers/ \
  --queries "Claude prompt engineering" "LoRA fine-tuning" \
  --reddit MachineLearning LocalLLaMA \
  --github karpathy/autoresearch anthropics/anthropic-sdk-python \
  --output-dir data/
```

### Key design decisions
- Async-first (`aiohttp`) for web scraping — 10x faster than sync
- Graceful degradation: missing deps (PyMuPDF, google-api) don't crash
- Dedup by SHA1(url+title) — avoids re-processing same pages across runs
- Chunking with overlap preserves context across paragraph boundaries

---

## ✅ Phase 2 — Remote Execution API (SHIPPED)

**Goal:** Push research jobs to Kaggle and poll results without manual intervention.

### What was built
- `api-triggers/trigger_kaggle.py` — generates + pushes Kaggle notebooks
- `api-triggers/poll_results.py` — polls + downloads + git-syncs
- `.github/workflows/research.yml` — `workflow_dispatch` + scheduled cron
- Auto-generated `kernel-metadata.json` with GPU + internet enabled

### Usage
```bash
# One-shot: push, wait, download
python api-triggers/trigger_kaggle.py \
  --topic "Claude skills optimization" \
  --iterations 20 \
  --github-repo yourusername/ultimate-doc-researcher \
  --download-results

# Or via GitHub Actions UI:
# Actions → "UltimateDocResearcher" → Run workflow → set topic + iterations
```

### Required GitHub Secrets
| Secret | Description |
|--------|-------------|
| `KAGGLE_USERNAME` | Your Kaggle username |
| `KAGGLE_API_TOKEN` | Kaggle API token |
| `GITHUB_TOKEN` | Auto-provided by Actions |
| `OPENAI_API_KEY` | Optional — for LLM Q&A generation |
| `GOOGLE_API_KEY` | Optional — for Google CSE |
| `GOOGLE_CX` | Optional — CSE engine ID |

---

## ✅ Phase 3 — Research Templates (SHIPPED)

**Goal:** Parameterised research programs that guide the autoresearch loop.

### What was built
- `templates/program_templates.py` — 4 built-in programs + dynamic generator
- `templates/program.md` — default program (Claude skills optimizer)

### Built-in programs
| Name | Topic |
|------|-------|
| `claude-skills-optimizer` | Anthropic Claude prompt/skills patterns |
| `mcp-agent-orchestration` | MCP tool design + multi-agent coordination |
| `openclaw-production` | Claude API at scale (batching, caching, cost) |
| `local-llm-fine-tuning` | LoRA/QLoRA on T4/3090 |

### Usage
```bash
# List programs
python templates/program_templates.py --list

# Generate program.md for a specific program
python templates/program_templates.py \
  --program claude-skills-optimizer \
  --output templates/program.md
```

---

## 🔄 Phase 4 — Integration & Polish (IN PROGRESS)

**Goal:** End-to-end demo, Docker, CI tests, production README.

### Remaining tasks
- [ ] `demo/demo.ipynb` — full walkthrough notebook
- [ ] `Dockerfile` — reproducible local dev environment
- [ ] `pyproject.toml` — proper Python packaging
- [ ] CI: `pytest` on collector unit tests
- [ ] README with architecture diagram + video demo link
- [ ] Results visualisation in `demo/results_viz.ipynb`

---

## 🚀 Phase 5 — Advanced Features (PLANNED)

### 5.1 Iterative corpus expansion
On each autoresearch iteration, generate new search queries from the
model's "knowledge gaps" and re-collect. The model's uncertainty on
val questions drives the next collection.

```python
# Pseudocode
for iteration in range(n):
    val_wrong = evaluate(model, val_set)          # questions model got wrong
    new_queries = generate_queries(val_wrong)     # LLM generates search terms
    new_docs = collector.collect(queries=new_queries)
    all_docs += new_docs
    retrain(model, all_docs)
```

### 5.2 Multi-model ensemble
Run the loop on 2–3 base models (Llama-3.2-3B, Phi-3.5-mini, Qwen2.5-3B)
and merge adapters with `mergekit` for a stronger final model.

### 5.3 Reward model scoring
Replace heuristic Q&A quality scoring with a small reward model trained
on human preference data (RLHF-lite).

### 5.4 Streaming results dashboard
FastAPI + SSE endpoint that streams results.tsv updates in real-time,
displayed in a simple React dashboard.

---

## Agent Communication Protocol

When running multiple agents in parallel (e.g., Collect + Train simultaneously
on different topics), they communicate via the file system:

```
data/
  lock/          # file locks (prevent concurrent writes to all_docs.txt)
  queue/         # JSON files: {"topic": "...", "status": "pending|running|done"}
  {topic_hash}/  # per-topic data isolation
    all_docs.txt
    train.jsonl
    val.jsonl
```

Agent heartbeat: each agent writes `data/lock/{pid}.heartbeat` every 30s.
Stale locks (>5min) are auto-cleared by the orchestrator.

---

## Performance Benchmarks (T4 GPU, Kaggle)

| Model | LoRA r | Batch | VRAM | Time/epoch | Val loss |
|-------|--------|-------|------|------------|----------|
| Llama-3.2-3B | 16 | 2×4 | 12GB | ~18min | ~1.8 |
| Llama-3.2-3B | 8 | 4×4 | 10GB | ~15min | ~1.9 |
| Phi-3.5-mini | 16 | 2×4 | 14GB | ~22min | ~1.7 |

*Benchmarks approximate — vary with dataset size and sequence length.*
