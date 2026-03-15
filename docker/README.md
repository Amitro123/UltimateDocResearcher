# Docker setup

Run the full stack with one command. No API keys required (Ollama is included).

```bash
# From the project root:
docker compose -f docker/docker-compose.yml up
```

Then open http://localhost:8501.

## First-time setup (one-time, ~2 GB download)

```bash
# Pull the default LLM (llama3.2 ~2 GB):
docker compose -f docker/docker-compose.yml run --rm ollama-pull
```

The model is stored in a named Docker volume (`udr-ollama-models`) and
survives `docker compose down`. You only need to pull it once.

## Services

| Service | URL | Description |
|---------|-----|-------------|
| `dashboard` | http://localhost:8501 | Run history, metrics, Research Packages |
| `chat` | http://localhost:8503 | Interactive research chat |
| `ollama` | http://localhost:11434 | Local LLM API |

## Running pipeline commands

```bash
# Run any pipeline command inside the app container:
docker compose -f docker/docker-compose.yml run --rm app \
  python -m autoresearch.research --topic "multi-tenant RAG"

# Collect documents:
docker compose -f docker/docker-compose.yml run --rm app \
  python -m collector.ultimate_collector --pdf-dir papers/ --output-dir data/

# Check LLM connectivity:
docker compose -f docker/docker-compose.yml run --rm app \
  python -m autoresearch.llm_client check
```

Collected documents and results are written to `../data/` and `../results/`
on your host machine (mounted as volumes).

## API keys (optional)

Copy `.env.example` to `.env` and add any keys you want to use.
Ollama always works without keys. Keys unlock higher-quality models:

```bash
cp .env.example .env
# Edit .env and add GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY
```

## Stop / reset

```bash
# Stop all services (data is preserved in volumes and bind mounts)
docker compose -f docker/docker-compose.yml down

# Full reset including downloaded models:
docker compose -f docker/docker-compose.yml down -v
```
