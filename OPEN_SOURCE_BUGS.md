# Open-Source Hardening Requirements for UltimateDocResearcher

This document outlines structural architectural flaws discovered during end-to-end evaluations of the Phase 13 single-command runner with the Gemini 2.5 Flash API. To achieve true open-source reliability, these bugs must be fixed at the foundation rather than using "lazy" workarounds.

## 1. The Prompt Cache Collision Bug
**The Bug:** The pipeline generated an `ARCHITECTURE.md` perfectly. Immediately afterward, the `code_suggester.py` step ran to generate code patterns, but it bizarrely output the exact text of the `ARCHITECTURE.md` file instead of Python code.
**The Root Cause:** The codebase uses `memory/cache.py` with a fuzzy Cosine Similarity layer (`>= 0.92`) to save LLM API costs. Because the bulk of the LLM prompt is the 40,000-character `corpus_extract` (the `user` prompt), the semantic embedding model fails to distinguish between the prompt requesting *Architecture* and the prompt requesting *Code*.
**Required Architectural Fix:**
- The embedding chunk generated for the cache key in `memory/cache.py` or `autoresearch/llm_client.py` MUST explicitly concatenate the `system` behavior prompt alongside the `user` context prompt before generating the vector embedding. 
- Alternatively, disable fuzzy caching by default when running vastly different prompts on the exact same large corpus text blocks.

## 2. The Silent API Truncation Bug
**The Bug:** The `gemini-2.5-flash` API consistently stopped generating code halfway through a `.md` Python code block (cutting off mid-word at exactly ~250 tokens), yet the pipeline saved the broken document and happily exited with "Exit Code 0 (Success)".
**The Root Cause:** Generating realistic Python code that touches the filesystem (`os`, `Path`) frequently triggers the strict default safety filters in the `google-genai` API (e.g., `HARM_CATEGORY_DANGEROUS_CONTENT`). When triggered, the API severs the stream, resulting in abrupt output cut-offs.
**Required Architectural Fix:**
- In `autoresearch/llm_client.py` (`_chat_google`), explicitly pass `safety_settings` configured to `BLOCK_NONE` within the `GenerateContentConfig` to prevent false-positive blocks on completely safe code-generation tasks.

## 3. Lack of Structural Delivery Validation
**The Bug:** The orchestrator blindly accepted and saved a deeply broken, mid-sentence truncated Markdown file to disk, scoring poorly (`2.0/5.0`) on downstream quality evaluations, instead of recognizing the network/API truncation.
**The Root Cause:** There are no safeguards enforcing output formatting reliability in `research_deliverables/generators.py` or `code_suggester.py`.
**Required Architectural Fix:**
- Implement a lightweight structural validator on LLM outputs before saving them to disk. For example, if the prompt mandates Markdown ````python` blocks, the system must assert that all opened code blocks are properly closed.
- If a truncation or invalid structure is detected, the script must throw an internal exception, flush the cache for that specific prompt, and trigger an automatic retry sequence before returning a success code.
