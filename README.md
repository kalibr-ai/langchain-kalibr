# langchain-kalibr

Execution path routing for AI agents. Kalibr routes your agents around failing models, tools, and configurations — before users notice.

[![PyPI](https://img.shields.io/pypi/v/langchain-kalibr)](https://pypi.org/project/langchain-kalibr/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-kalibr)](https://pypi.org/project/langchain-kalibr/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What is this?

**`langchain-kalibr`** is a LangChain integration that gives your chains and agents adaptive model routing. Instead of hardcoding a single LLM provider, you define multiple models (paths) and Kalibr learns which one works best for each task — then routes traffic accordingly.

- **Drop-in replacement** for `ChatOpenAI`, `ChatAnthropic`, etc.
- **Works with LangChain chains, agents, and LangGraph**
- **Works with CrewAI** (accepts any LangChain LLM)
- **Automatic fallback** — if one model fails, tries the next
- **Outcome learning** — report success/failure and Kalibr improves routing over time

## Installation

```bash
pip install langchain-kalibr
```

With specific provider support:

```bash
pip install langchain-kalibr[openai]      # OpenAI models
pip install langchain-kalibr[anthropic]   # Anthropic models
pip install langchain-kalibr[all]         # All providers
```

## Setup

Get your credentials from [dashboard.kalibr.systems/settings](https://dashboard.kalibr.systems/settings):

```bash
export KALIBR_API_KEY="your-api-key"
export KALIBR_TENANT_ID="your-tenant-id"
export OPENAI_API_KEY="sk-..."            # for OpenAI models
export ANTHROPIC_API_KEY="sk-ant-..."     # for Anthropic models
export GOOGLE_API_KEY=...                # for Gemini models
```

## Quick Start

```python
from langchain_kalibr import ChatKalibr

# Define models to route between
llm = ChatKalibr(
    goal="summarize",
    paths=["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
)

# Use like any LangChain chat model
response = llm.invoke("Summarize the key benefits of adaptive routing.")
print(response.content)

# Report outcome to improve future routing
llm.report(success=True)
```

## Use in LangChain Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_kalibr import ChatKalibr

llm = ChatKalibr(
    goal="answer_questions",
    paths=["gpt-4o", "claude-sonnet-4-20250514", "gpt-4o-mini"],
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"question": "What is Thompson Sampling?"})
```

## Use with LangGraph Agents

```python
from langgraph.prebuilt import create_react_agent
from langchain_kalibr import ChatKalibr

llm = ChatKalibr(
    goal="agent_tasks",
    paths=["gpt-4o", "claude-sonnet-4-20250514"],
)

agent = create_react_agent(llm, tools=[...])
result = agent.invoke({"messages": [("human", "Search for recent AI news")]})
```

## Use with CrewAI

CrewAI accepts any LangChain LLM natively:

```python
from crewai import Agent, Task, Crew
from langchain_kalibr import ChatKalibr

llm = ChatKalibr(
    goal="research",
    paths=["gpt-4o", "claude-sonnet-4-20250514"],
)

researcher = Agent(
    role="Research Analyst",
    goal="Find and summarize key information",
    llm=llm,
)

task = Task(
    description="Research the latest developments in adaptive AI routing.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## Outcome Reporting

Outcome reporting is what makes Kalibr learn. After each call, tell Kalibr whether it worked:

### Manual Reporting

```python
response = llm.invoke("Extract the email from: Contact us at hello@example.com")

# Check if the task succeeded
has_email = "@" in response.content
llm.report(success=has_email)
```

### Auto-Reporting with `success_when`

```python
llm = ChatKalibr(
    goal="extract_email",
    paths=["gpt-4o", "claude-sonnet-4-20250514"],
    success_when=lambda output: "@" in output,
)

# Outcome reported automatically after each call
response = llm.invoke("Extract the email from: Contact us at hello@example.com")
```

## Advanced: Routing Between Configurations

Route between different parameter configurations, not just models:

```python
llm = ChatKalibr(
    goal="creative_writing",
    paths=[
        {"model": "gpt-4o", "params": {"temperature": 0.3}},
        {"model": "gpt-4o", "params": {"temperature": 0.9}},
        {"model": "claude-sonnet-4-20250514", "params": {"temperature": 0.7}},
    ],
)
```

Route between different tool configurations:

```python
llm = ChatKalibr(
    goal="research",
    paths=[
        {"model": "gpt-4o", "tools": ["web_search"]},
        {"model": "gpt-4o", "tools": ["code_interpreter"]},
        {"model": "claude-sonnet-4-20250514"},
    ],
)
```

## How Routing Works

**Trust invariant:** Success rate always dominates. Cost and latency only break ties between paths with comparable success rates. Kalibr never sacrifices quality for cost savings.

1. **You define paths** — models (+ optional tools/params) that can handle your task
2. **Kalibr picks** — uses Thompson Sampling to balance trying new options vs. using what works
3. **You report outcomes** — tell Kalibr if the task succeeded
4. **Kalibr learns** — routes more traffic to what works, automatically routes around degradation

Kalibr explores with ~10% canary traffic to continuously discover better options, while routing the majority of traffic to the best-performing path.

## API Reference

### `ChatKalibr`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `goal` | `str` | required | Task name for routing |
| `paths` | `list` | `["gpt-4o"]` | Models or configs to route between |
| `api_key` | `str` | env var | Kalibr API key |
| `tenant_id` | `str` | env var | Kalibr tenant ID |
| `success_when` | `callable` | `None` | Auto-evaluate success |
| `exploration_rate` | `float` | `None` | Override exploration rate (0.0-1.0) |

### Methods

| Method | Description |
|---|---|
| `invoke(messages)` | Send messages, get routed response |
| `report(success, reason, score)` | Report outcome for routing improvement |
| `last_trace_id` | Get trace ID from last call |
| `last_model_id` | Get which model was used last |

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `KALIBR_API_KEY` | API key from dashboard | Yes |
| `KALIBR_TENANT_ID` | Tenant ID from dashboard | Yes |
| `OPENAI_API_KEY` | OpenAI API key | If using OpenAI models |
| `ANTHROPIC_API_KEY` | Anthropic API key | If using Anthropic models |
| `GOOGLE_API_KEY` | Google API key | If using Google models |

## Links

- [Kalibr Dashboard](https://dashboard.kalibr.systems)
- [Kalibr Docs](https://kalibr.systems/docs)
- [Kalibr Python SDK](https://pypi.org/project/kalibr/)
- [GitHub](https://github.com/kalibr-ai/langchain-kalibr)
- [LangChain Docs](https://docs.langchain.com)

## License

MIT
