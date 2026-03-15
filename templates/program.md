# Research Program: UltimateDocResearcher-Reliability

## Topic
LLM evaluation best practices

## Objective
Analyze the current codebase, historical bugs (Issue #18), and research results to identify gaps in reliability, performance, and API handling. Propose concrete code improvements for the research pipeline.

## Evaluation Criteria
- Alignment with "Phase 4: Integration & Polish" goals
- Resolution of identified bugs in CODE_REVIEW.md
- Robustness of the Gemini integration and retry logic
- Completeness of the end-to-end research flow

## Key Research Questions
1. How can we improve the robustness of the Gemini API client (beyond simple retries)?
2. What are the most critical architectural gaps identified in the "Comprehensive Code Review"?
3. How can we optimize the corpus cleaning and chunking logic for meta-documentation?
4. What patterns can we extract from "Issue #18" to prevent regressions in evaluation?
5. How does the E2E guide verify the correctness of the final code suggestions?

## Preferred Sources
- local: CODE_REVIEW.md
- local: UltimateDocResearcher_CR.md
- local: bug_analysis_eval.md
- local: AGENTS.md

## Output Format
Structured code suggestions and architectural improvement plan.

## Model
gemini-2.5-flash-lite
