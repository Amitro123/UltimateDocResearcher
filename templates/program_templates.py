"""
templates/program_templates.py
--------------------------------
Generates research program definitions (program.md files) for different topics.
These get injected into the autoresearch loop as the "research objective".

A program.md tells the model:
  - What to study
  - What questions to answer
  - What evaluation metric to optimise
  - What outputs to produce
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROGRAMS = {}  # registry


@dataclass
class ResearchProgram:
    name: str
    topic: str
    objective: str
    evaluation_criteria: List[str]
    key_questions: List[str]
    output_format: str
    preferred_sources: List[str] = field(default_factory=list)
    model_hint: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

    def to_markdown(self) -> str:
        criteria_md = "\n".join(f"- {c}" for c in self.evaluation_criteria)
        questions_md = "\n".join(f"{i+1}. {q}" for i, q in enumerate(self.key_questions))
        sources_md = "\n".join(f"- {s}" for s in self.preferred_sources) or "- Any relevant source"

        return f"""\
# Research Program: {self.name}

## Topic
{self.topic}

## Objective
{self.objective}

## Evaluation Criteria
{criteria_md}

## Key Research Questions
{questions_md}

## Preferred Sources
{sources_md}

## Output Format
{self.output_format}

## Model
{self.model_hint}
"""

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


def register(program: ResearchProgram) -> ResearchProgram:
    PROGRAMS[program.name] = program
    return program


# ── Built-in programs ──────────────────────────────────────────────────────────

register(ResearchProgram(
    name="claude-skills-optimizer",
    topic="Anthropic Claude skills and prompt optimization",
    objective=(
        "Discover the most effective patterns for writing Claude system prompts, "
        "skills, and multi-turn conversation structures. Identify which instruction "
        "formats lead to highest task completion and alignment."
    ),
    evaluation_criteria=[
        "Task completion rate on held-out benchmarks",
        "Instruction-following precision",
        "Reduction in hallucinations",
        "Efficiency (tokens used per correct answer)",
    ],
    key_questions=[
        "What prompt structures produce the most reliable tool-use in Claude?",
        "How do XML tags vs. markdown headers affect Claude's parsing accuracy?",
        "What is the optimal system prompt length for complex agentic tasks?",
        "How does few-shot exemplar order affect Claude's output quality?",
        "What patterns reduce sycophancy in Claude's responses?",
    ],
    output_format="A fine-tuned LoRA adapter + structured prompt template library in JSON",
    preferred_sources=[
        "https://docs.anthropic.com",
        "r/ClaudeAI",
        "github.com/anthropics",
        "arxiv.org (prompt engineering)",
    ],
))

register(ResearchProgram(
    name="mcp-agent-orchestration",
    topic="MCP (Model Context Protocol) agent orchestration patterns",
    objective=(
        "Build a comprehensive understanding of MCP tool design, "
        "multi-agent coordination, and failure recovery patterns. "
        "Produce a fine-tuned model that can design MCP tool schemas."
    ),
    evaluation_criteria=[
        "Schema validity rate",
        "Tool invocation success rate on synthetic tasks",
        "Error recovery without human intervention",
    ],
    key_questions=[
        "What MCP tool schema patterns are most reliably parsed by LLMs?",
        "How should multi-agent handoffs be structured to minimise context loss?",
        "What are common failure modes in long agentic MCP chains?",
        "How does parallelism in tool calls affect overall latency?",
        "What is the best way to handle streaming results in MCP?",
    ],
    output_format="LoRA adapter + MCP schema cookbook (JSON + MD)",
    preferred_sources=[
        "github.com/modelcontextprotocol",
        "r/LocalLLaMA",
        "arxiv.org (tool-use LLM)",
    ],
))

register(ResearchProgram(
    name="openclaw-production",
    topic="OpenClaw production patterns for Claude API at scale",
    objective=(
        "Identify production-grade patterns for building high-throughput, "
        "fault-tolerant Claude API applications. Focus on batching, caching, "
        "rate-limit handling, and cost optimisation."
    ),
    evaluation_criteria=[
        "Throughput (requests/second)",
        "Cost per 1000 successful completions",
        "P99 latency",
        "Error rate under load",
    ],
    key_questions=[
        "How should prompt caching be structured to maximise cache hit rate?",
        "What batching strategies work best for async Claude API calls?",
        "How to implement graceful degradation when hitting rate limits?",
        "What monitoring metrics are most predictive of degraded quality?",
        "How do tool_choice strategies affect latency vs. accuracy tradeoffs?",
    ],
    output_format="LoRA adapter + production runbook in Markdown",
    preferred_sources=[
        "docs.anthropic.com/api",
        "r/MachineLearning",
        "github.com/anthropics/anthropic-sdk-python",
    ],
))

register(ResearchProgram(
    name="local-llm-fine-tuning",
    topic="Local LLM fine-tuning with LoRA/QLoRA on consumer hardware",
    objective=(
        "Develop best practices for efficient fine-tuning of 3B-13B parameter "
        "models on single-GPU setups (T4, A10G, 3090). Focus on hyperparameter "
        "selection, data curation, and evaluation."
    ),
    evaluation_criteria=[
        "Val loss convergence speed",
        "VRAM usage",
        "Training time per epoch",
        "Downstream task accuracy delta vs base model",
    ],
    key_questions=[
        "What LoRA rank produces the best quality/VRAM tradeoff for 3B models?",
        "How does data quality vs. quantity affect fine-tuning outcomes?",
        "What learning rate schedule works best for short fine-tuning runs?",
        "How to detect and prevent catastrophic forgetting?",
        "What evaluation benchmarks are most correlated with real-world utility?",
    ],
    output_format="LoRA adapter + hyperparameter sweep results in TSV",
    preferred_sources=[
        "github.com/unslothai/unsloth",
        "r/LocalLLaMA",
        "arxiv.org (LoRA, QLoRA)",
        "huggingface.co/blog",
    ],
))


# ── Dynamic program generator ──────────────────────────────────────────────────

def create_program(
    topic: str,
    objective: Optional[str] = None,
    questions: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> ResearchProgram:
    """Auto-generate a program.md for any arbitrary topic."""
    return ResearchProgram(
        name=topic.lower().replace(" ", "-")[:40],
        topic=topic,
        objective=objective or (
            f"Build deep expertise in {topic} by collecting and synthesising "
            f"the best available literature, code examples, and practitioner wisdom."
        ),
        evaluation_criteria=[
            "Factual accuracy on held-out Q&A pairs",
            "Coverage breadth across subtopics",
            "Source diversity",
        ],
        key_questions=questions or [
            f"What are the foundational concepts in {topic}?",
            f"What are the most common misconceptions about {topic}?",
            f"What are the current state-of-the-art approaches to {topic}?",
            f"What are the practical limitations of {topic}?",
            f"What open problems remain in {topic}?",
        ],
        output_format="LoRA adapter + structured knowledge base",
        preferred_sources=sources or ["arxiv.org", "r/MachineLearning"],
    )


def get_program(name: str) -> ResearchProgram:
    if name in PROGRAMS:
        return PROGRAMS[name]
    return create_program(topic=name)


def list_programs() -> List[str]:
    return list(PROGRAMS.keys())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", default="claude-skills-optimizer")
    parser.add_argument("--output", default="templates/program.md")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name in list_programs():
            print(f"  • {name}")
    else:
        prog = get_program(args.program)
        path = prog.save(args.output)
        print(f"✅ Program saved: {path}")
        print(prog.to_markdown())
