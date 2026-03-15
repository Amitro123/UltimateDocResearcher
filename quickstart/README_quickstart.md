# Quickstart — 2 minutes to first output

No API key. No GPU. No PDF downloads. Just Python 3.10+.

```bash
git clone https://github.com/Amitro123/UltimateDocResearcher
cd ultimate-doc-researcher
pip install -r requirements.txt
make quickstart
```

That's it. In ~2 minutes you'll have:
- `results/quickstart-code-suggestions.md` — 3 copy-paste Python patterns
- `results/quickstart-demo/SUMMARY.md` — executive overview
- `results/quickstart-demo/IMPLEMENTATION.md` — step-by-step plan
- `results/quickstart-demo/NEXT_STEPS.md` — prioritised actions

---

## What the demo uses

The demo runs on a **bundled synthetic corpus** (`quickstart/demo-corpus.txt`)
about Claude tool use and SDK patterns. It does not need:
- Internet access
- Any API key
- Ollama installed
- Any PDFs

If Ollama or an API key is available, the demo will use it automatically for
higher-quality output. Without either, it falls back to a heuristic mode that
still produces useful (if lower-quality) output.

---

## From demo → your own topic

Once the demo works, research your own topic:

```bash
# 1. Add your PDFs to papers/
cp ~/Downloads/my-paper.pdf papers/

# 2. Reset workspace for the new topic
python new_run.py --topic "my research topic"

# 3. Collect + generate
python -m collector.ultimate_collector --pdf-dir papers/ --output-dir data/
python -m autoresearch.research --topic "my research topic"
```

Or use the interactive chat interface:
```bash
make chat
# open http://localhost:8503
```

---

## What if something fails?

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Output is very short / generic | Install Ollama (`ollama pull llama3.2`) or set `GOOGLE_API_KEY` |
| `FileNotFoundError: demo-corpus.txt` | `python quickstart/build_demo_corpus.py` |
| Port 8501 already in use | `make dashboard` uses `--server.port 8501` — change in Makefile |

See the full [Troubleshooting section](../E2E_GUIDE.md#troubleshooting) in E2E_GUIDE.md.

---

## Full Docker setup (optional)

If you want the full stack (Ollama + dashboard + chat) in one command:

```bash
make docker-up
# First time: make docker-up && docker compose -f docker/docker-compose.yml run --rm ollama-pull
```

See [docker/README.md](../docker/README.md) for details.
