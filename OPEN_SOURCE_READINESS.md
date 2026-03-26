# Open Source Readiness Assessment — UltimateDocResearcher

> Assessment date: 2026-03-27
> Assessed by: Claude Sonnet 4.6 (after full E2E verification + bug-fix session)

---

## Overall Score: 7.2 / 10

The project is **ready to publish after resolving 4 blockers**. The core pipeline is
genuinely well-built, tests are real, documentation covers operational pitfalls, and the
end-to-end research loop works.

---

## Scorecard

| Area | Score | Notes |
|------|-------|-------|
| **Architecture** | 9/10 | Clean separation: collector → analyze → prepare → generate → eval. Async-first I/O, incremental dedup, prompt caching, run memory. |
| **Functionality** | 8/10 | Full pipeline works end-to-end. Multiple LLM backends (Ollama, Gemini, OpenAI, Anthropic). Heuristic fallbacks when LLM unavailable. Docker support. |
| **Testing** | 7.5/10 | 347 tests, 16 test modules, 0 warnings, all pass. Good mock discipline. Covers scraper edge cases (403, HTML responses, asyncio event loop scenarios). |
| **Documentation** | 7/10 | README + E2E_GUIDE + CLAUDE.md + AGENTS.md is more than most OSS projects ship. Mermaid architecture diagram. Quickstart. |
| **Packaging** | 7/10 | PEP 517 compliant, correct build backend (`setuptools.build_meta`), all packages discoverable, 4 CLI entry points, ruff configured. |
| **CI/CD** | 6/10 | GitHub Actions workflow exists for Kaggle kernel dispatch — but no `pytest` job. Tests don't run automatically on push/PR. |
| **Code Quality** | 7/10 | Consistent async patterns, retry/backoff baked in, UTF-8 encoding handled across all entry points, user-facing error messages. |

---

## Bugs Fixed in This Session

All 4 GitHub issues from `issues.md` were resolved, plus additional bugs found during E2E verification:

| # | Issue | Fix |
|---|-------|-----|
| 1 | Wrong build backend in `pyproject.toml` | Changed `setuptools.backends.legacy:build` → `setuptools.build_meta` |
| 2 | Incomplete package discovery + missing runtime deps | Added 5 packages, 5 runtime deps to `pyproject.toml` |
| 3 | Unawaited coroutine in `collector/scraper.py` | `pool.submit(asyncio.run, _run())` → `pool.submit(lambda: asyncio.run(_run()))` |
| 4 | Missing IMPLEMENTATION.md / NEXT_STEPS.md deliverables | Restored to `code` and `process` types in `classify_topic.py` |
| 5 | Unicode/emoji crash on Windows cp1252 | Added UTF-8 `reconfigure` to 14 entry scripts + 2 library modules |
| 6 | `PermissionError` when deleting locked `prompts.db` | Wrapped `unlink()` in `try/except PermissionError` with user warning |
| 7 | `asyncio.run()` on sync `UltimateCollector.run()` | Removed `asyncio.run()` wrapper, call directly |
| 8 | `read_text()` without encoding in `api-triggers/` | Added `encoding="utf-8"` to all `read_text()` calls |
| 9 | `open()` without encoding in README + E2E_GUIDE docs | Added `encoding='utf-8'` to all 4 code snippets in docs |
| 10 | Unawaited coroutine warnings in test mocks | Added `coro.close()` helper to 3 test cases |

---

## Blockers — Must Fix Before Publishing

### 1. No `LICENSE` file
`pyproject.toml` declares `license = {text = "MIT"}` but there is no `LICENSE` file in
the repository. GitHub won't show the license badge and forks are legally ambiguous
without it.

**Fix:** Create a standard `LICENSE` file with MIT text.

---

### 2. `results/` directory contains personal research artifacts
The `results/` directory has 27+ research run outputs from development sessions. These
are not part of the open source project and will confuse contributors.

**Fix:** Add `results/` to `.gitignore`. If any results are already committed, clean
them from git history.

---

### 3. No `pytest` job in CI
`.github/workflows/research.yml` only dispatches Kaggle kernels. There is no workflow
that runs `pytest` on pull requests. Contributors get no automated test feedback.

**Fix:** Create `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
```

---

### 4. No `CONTRIBUTING.md`
Standard open source expectation. Without it, contributors don't know how to set up a
dev environment, run tests, or submit PRs.

**Fix:** Create `CONTRIBUTING.md` covering:
- Dev environment setup (`python -m venv .venv`, `pip install -e .`)
- Running tests (`pytest tests/`)
- Running the E2E pipeline
- PR/commit conventions

---

## Should Fix Before Publishing

### 5. `.env` with real API key in project directory
The `.env` file is not git-tracked (confirmed), but it contains a real `GOOGLE_API_KEY`.
Rotate this key in Google Cloud Console as a precaution before any public activity on
this repo.

Ship only `.env.example` and add a setup step: `cp .env.example .env`.

---

### 6. Generated tracking files in repo root
`issues.json` and `issues.md` are generated artifacts from the development session.
Add them to `.gitignore` or delete before publishing.

---

### 7. Python version discrepancy
README says "Python 3.10+" but `pyproject.toml` says `requires-python = ">=3.11"`.
Pick one and be consistent. (3.11 is recommended — `tomllib` stdlib, better error messages.)

---

### 8. `data/` and `results/` in `.gitignore`
Verify both are excluded so users don't accidentally commit research artifacts.

---

## Nice to Have

| Item | Notes |
|------|-------|
| `SECURITY.md` | Security disclosure policy — expected by GitHub community health score for projects handling API keys |
| Test coverage report | Add `pytest-cov` to dev deps, report in CI. Currently no visibility into % coverage |
| Pin `v0.1.0` in CHANGELOG | `CHANGELOG.md` exists but has no versioned release entry yet |
| Validate quickstart in CI | The `demo/` quickstart demo exists but isn't tested in CI — it will bit-rot |

---

## Minimum Publish Checklist

```
[ ] Create LICENSE file (MIT)
[ ] Add results/, issues.json, issues.md to .gitignore
[ ] Create .github/workflows/test.yml (pytest on push/PR)
[ ] Create CONTRIBUTING.md (dev setup + test instructions)
[ ] Fix Python version consistency: README vs pyproject.toml (use 3.11)
[ ] Rotate the Google API key from .env (precaution)
```

---

## Verdict

**Publish after the 6 checklist items above.** The project is more production-ready than
the average open source research tool. The architecture is sound, the tests are
meaningful (not just smoke tests), the documentation covers real operational pitfalls
(Windows encoding, SQLite locking, rate limits, asyncio event loop edge cases), and the
full research pipeline was verified end-to-end. The blockers are all mechanical — under
2 hours of work combined.
