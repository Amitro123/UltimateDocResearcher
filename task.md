Phase 12: Dynamic Templates by Input Type

Current: Fixed templates fail on error logs

Goal: classify_input() → input_type → relevant templates only

Types + templates:
error_log → ROOT_CAUSE.md + FIX_STEPS.md + PREVENTION.md
codebase → CODE/ + ARCHITECTURE.md + TESTS.md
paper → SUMMARY.md + KEY_TAKEAWAYS.md + BENCHMARKS.md
website → FLOW.md + INTEGRATION.md

Types + templates:
text

Implementation:
research_deliverables/classify_input.py
templates/error_log/ + paper/ + etc.
CLI: python -m autoresearch.research --input-type error_log

Test: Vertex AI log → remediation package!

Start with classify_input() → dynamic template loader!
