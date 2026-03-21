# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
### Fixed
- **BUG-01**: Fixed `AttributeError` in `autoresearch/research.py` where `mem.complete_run()` was called instead of `mem.finish_run()`.
- **tests/test_llm_client.py**: Fixed fallback tests that were failing due to un-patched `GOOGLE_API_KEY`.

### Changed
- **CODE_REVIEW.md**: Replaced with a more comprehensive and detailed code review report.
- **README.md**: Updated project structure to include `research_deliverables/` module and added notes on Gemini model prioritization.
