# UltimateDocResearcher Issues

## [Issue #7] Packaging metadata omits required modules and runtime dependencies, making `pip install .` incomplete
**State:** open | **Created:** 2026-03-24T21:11:43Z
**Link:** https://github.com/Amitro123/UltimateDocResearcher/issues/7

## Summary
Even after the build-backend problem is fixed, the published package metadata is still incomplete.

Two problems are visible directly in `pyproject.toml`:

1. package discovery only includes `collector*`, `autoresearch*`, `api-triggers*`, and `templates*`
2. `[project.dependencies]` omits several libraries that are imported by the codebase and are present in `requirements.txt`

Relevant files:

- package selection: https://github.com/Amitro123/UltimateDocResearcher/blob/9ea1194/pyproject.toml#L34-L36
- declared dependencies: https://github.com/Amitro123/UltimateDocResearcher/blob/9ea1194/pyproject.toml#L13-L26
- requirements.txt: https://github.com/Amitro123/UltimateDocResearcher/blob/9ea1194/requirements.txt

## Missing packages from wheel discovery
Top-level packages in the repo include:

- `chat`
- `dashboard`
- `eval`
- `memory`
- `research_deliverables`

but those are not included in the setuptools discovery config.

That means a wheel built from this project would omit code that is referenced by the CLI and documented workflows.

## Missing runtime dependencies
`requirements.txt` and the codebase show runtime usage of libraries that are not declared in `[project.dependencies]`, including:

- `anthropic` (`autoresearch/llm_client.py`)
- `google-genai` (`autoresearch/llm_client.py`)
- `streamlit` (`chat/app.py`, `dashboard/app.py`)
- `pyyaml` (`collector/analyzer.py`, `autoresearch/code_suggester.py`)
- `jinja2` (`research_deliverables/generators.py`)

## Why this matters
A user who follows the modern Python path (`pip install .`, wheel install, or PyPI install) will get an environment that is materially different from `pip install -r requirements.txt`.

That leads to import errors, missing CLI functionality, and packaging surprises in CI.

## Expected behavior
The wheel/sdist metadata should match the actual runnable project:

- all required top-level packages included
- all runtime imports declared in `[project.dependencies]`
- install-from-wheel behavior consistent with install-from-requirements

## Suggested fix
- expand package discovery to include all shipped Python packages
- move required runtime deps from `requirements.txt` into `[project.dependencies]`
- add a packaging smoke test in CI that installs the built wheel into a clean venv and imports the major entry points

---

## [Issue #6] `collector.scraper.scrape_topic()` creates unawaited coroutines in the sync wrapper path
**State:** open | **Created:** 2026-03-24T21:11:43Z
**Link:** https://github.com/Amitro123/UltimateDocResearcher/issues/6

## Summary
The synchronous `scrape_topic()` wrapper creates the `_run()` coroutine object before handing execution off to the thread-pool / `asyncio.run` path.

Relevant code: https://github.com/Amitro123/UltimateDocResearcher/blob/9ea1194/collector/scraper.py#L166-L176

Current implementation:

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    return pool.submit(asyncio.run, _run()).result()
```

## Why this is problematic
Because `_run()` is evaluated eagerly in the caller thread, you can end up with an unawaited coroutine when the execution path is mocked, short-circuited, or errors before `asyncio.run()` actually consumes it.

This already shows up in the current test suite as runtime warnings during `pytest`:

- `RuntimeWarning: coroutine 'scrape_topic.<locals>._run' was never awaited`

The same pattern also makes the sync wrapper harder to test reliably.

## Reproduction
Run the full test suite:

```bash
python -m pytest tests/ -q
```

Among the current failures/warnings, pytest reports unawaited coroutine warnings pointing at `collector/scraper.py:173`.

## Expected behavior
The coroutine should be created inside the worker / execution function, e.g. defer creation until the submitted callable actually runs.

Something along the lines of:

```python
pool.submit(lambda: asyncio.run(_run()))
```

or a small helper function would avoid eager coroutine creation.

## Suggested follow-up
After fixing the wrapper, tighten the scraper tests so they assert the no-warning behavior in both:

- normal sync context
- already-running event loop context

---

## [Issue #5] Topic deliverable classification dropped `IMPLEMENTATION.md` and `NEXT_STEPS.md`, contradicting docs and tests
**State:** open | **Created:** 2026-03-24T21:11:43Z
**Link:** https://github.com/Amitro123/UltimateDocResearcher/issues/5

## Summary
`research_deliverables/classify_topic.py` currently returns the same reduced deliverable set for all topic types:

- `SUMMARY.md`
- `ARCHITECTURE.md`
- `PLAN.md`
- `RISKS.md`
- `BENCHMARKS.md`

That means `IMPLEMENTATION.md` and `NEXT_STEPS.md` are never generated for normal text topics, even though the rest of the project still documents them as part of the package and the test suite expects them.

Relevant files:

- classifier implementation: https://github.com/Amitro123/UltimateDocResearcher/blob/9ea1194/research_deliverables/classify_topic.py#L125-L212
- package docs in `autoresearch/research.py`: https://github.com/Amitro123/UltimateDocResearcher/blob/9ea1194/autoresearch/research.py#L10-L19
- README output description: https://github.com/Amitro123/UltimateDocResearcher/blob/9ea1194/README.md#L102-L103
- failing tests: https://github.com/Amitro123/UltimateDocResearcher/blob/9ea1194/tests/test_classify_topic.py#L100-L126

## Reproduction
Run:

```bash
python -m pytest tests/test_classify_topic.py -q
```

Observed failures include:

- process topics no longer include `IMPLEMENTATION.md`
- code topics no longer include `NEXT_STEPS.md`

A full test run currently fails on these assertions as well.

## Why this matters
This is more than a docs mismatch:

- expected deliverables silently disappear from generated research packages
- downstream tooling/docs still advertise a richer package than users actually get
- the regression is already captured by the existing tests

## Expected behavior
Either:

1. restore `IMPLEMENTATION.md` / `NEXT_STEPS.md` to the canonical deliverable set, or
2. update docs/tests/generator expectations consistently if the product decision was to remove them

Right now the repository is in an inconsistent middle state.

---

## [Issue #4] pyproject.toml uses a non-existent setuptools build backend, so wheel builds fail immediately
**State:** open | **Created:** 2026-03-24T21:11:18Z
**Link:** https://github.com/Amitro123/UltimateDocResearcher/issues/4

## Summary
`pyproject.toml` currently declares `setuptools.backends.legacy:build` as the build backend. That backend does not exist, so standard PEP 517 builds fail before packaging even starts.

File reference: https://github.com/Amitro123/UltimateDocResearcher/blob/9ea1194/pyproject.toml#L1-L4

## Reproduction
From a fresh clone:

```bash
python -m pip install build
python -m build --wheel --no-isolation
```

Actual result:

```text
ERROR Backend 'setuptools.backends.legacy:build' is not available.
```

## Why this matters
This blocks the standard Python packaging/install flow for anyone trying to:

- build a wheel
- publish to PyPI/TestPyPI
- validate packaging in CI
- consume the project via normal PEP 517 tooling

## Expected behavior
`python -m build` should produce a wheel/sdist successfully.

## Likely fix
Use a valid setuptools backend such as one of:

- `setuptools.build_meta`
- `setuptools.build_meta:__legacy__`

and then add a packaging CI check so this regresses less easily.
