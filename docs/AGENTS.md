# AGENTS.md — Kalibr SDK

> This file helps AI coding agents understand how to use Kalibr.

## What is Kalibr?

Kalibr is an adaptive routing SDK for AI agents. Instead of hardcoding a single LLM provider, you define multiple models and Kalibr learns which one works best for each task — then routes automatically using Thompson Sampling.

## When to Use Kalibr

Use Kalibr when you need:
- **Model routing** — route between OpenAI, Anthropic, Google models
- **Automatic fallback** — if one provider fails, try the next
- **Cost optimization** — learn which cheaper model performs adequately
- **Outcome-based learning** — improve routing based on success/failure signals
- **Multi-model experimentation** — A/B test models with Thompson Sampling

## Installation

```bash
pip install kalibr                  # Core SDK
pip install langchain-kalibr        # LangChain integration
pip install kalibr[crewai]          # CrewAI integration
pip install kalibr[openai-agents]   # OpenAI Agents SDK integration
```

## Quick Start

```python
from kalibr import Router

router = Router(
    goal="extract_company",
    paths=["gpt-4o", "claude-sonnet-4-20250514"]
)

response = router.completion(
    messages=[{"role": "user", "content": "Extract the company: Hi, I'm Sarah from Stripe."}]
)

router.report(success=True)
```

## LangChain Integration

```python
from langchain_kalibr import ChatKalibr

llm = ChatKalibr(
    goal="summarize",
    paths=["gpt-4o", "claude-sonnet-4-20250514"],
)

# Use in any LangChain chain
chain = prompt | llm | parser
```

## CrewAI Integration

```python
from langchain_kalibr import ChatKalibr
from crewai import Agent

llm = ChatKalibr(goal="research", paths=["gpt-4o", "claude-sonnet-4-20250514"])
agent = Agent(role="Researcher", goal="Find information", llm=llm)
```

## Key Concepts

- **Goal**: A named task (e.g., "summarize", "extract_email")
- **Path**: A model + optional tools + optional params
- **Outcome**: Success/failure signal that teaches Kalibr
- **Thompson Sampling**: Algorithm that balances exploration vs exploitation
- **Canary traffic**: ~10% of requests explore new options

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| KALIBR_API_KEY | Yes | From dashboard.kalibr.systems |
| KALIBR_TENANT_ID | Yes | From dashboard.kalibr.systems |
| OPENAI_API_KEY | If using OpenAI | OpenAI API key |
| ANTHROPIC_API_KEY | If using Anthropic | Anthropic API key |

## Links

- PyPI: https://pypi.org/project/kalibr/
- LangChain: https://pypi.org/project/langchain-kalibr/
- Dashboard: https://dashboard.kalibr.systems
- Docs: https://kalibr.systems/docs
