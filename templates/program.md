# Research Program: claude-skills-optimizer

## Topic
Anthropic Claude skills and prompt optimization

## Objective
Discover the most effective patterns for writing Claude system prompts,
skills, and multi-turn conversation structures. Identify which instruction
formats lead to highest task completion and alignment.

## Evaluation Criteria
- Task completion rate on held-out benchmarks
- Instruction-following precision
- Reduction in hallucinations
- Efficiency (tokens used per correct answer)

## Key Research Questions
1. What prompt structures produce the most reliable tool-use in Claude?
2. How do XML tags vs. markdown headers affect Claude's parsing accuracy?
3. What is the optimal system prompt length for complex agentic tasks?
4. How does few-shot exemplar order affect Claude's output quality?
5. What patterns reduce sycophancy in Claude's responses?

## Preferred Sources
- https://docs.anthropic.com
- r/ClaudeAI
- github.com/anthropics
- arxiv.org (prompt engineering)

## Output Format
A fine-tuned LoRA adapter + structured prompt template library in JSON

## Model
unsloth/Llama-3.2-3B-Instruct-bnb-4bit
