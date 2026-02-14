# Adaptive Model Routing for CrewAI with Kalibr

Hey CrewAI community! 👋

Wanted to share an integration we built: **Kalibr** — adaptive model routing that learns which LLMs work best for your CrewAI agents.

## The Problem

When you're running a Crew, different agents often perform better with different models. Your researcher might nail it with GPT-4o, but your writer might do better with Claude. And model performance changes over time — what worked last month might not be optimal today.

## What Kalibr Does

Kalibr sits between your agents and LLM providers. It uses Thompson Sampling to:
- **Learn** which models work best for each task
- **Route** traffic to the best-performing model
- **Explore** with ~10% canary traffic to discover better options
- **Fallback** automatically if a provider goes down

## Working Example

```python
from crewai import Agent, Task, Crew
from langchain_kalibr import ChatKalibr

# Define routing — Kalibr learns which model works best
research_llm = ChatKalibr(
    goal="research",
    paths=["gpt-4o", "claude-sonnet-4-20250514"],
)

writing_llm = ChatKalibr(
    goal="writing",
    paths=["claude-sonnet-4-20250514", "gpt-4o"],
    success_when=lambda out: len(out) > 200,  # auto-report
)

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, comprehensive information",
    llm=research_llm,
)

writer = Agent(
    role="Content Writer",
    goal="Write clear, engaging content",
    llm=writing_llm,
)

research_task = Task(
    description="Research the latest developments in {topic}.",
    agent=researcher,
)

writing_task = Task(
    description="Write a summary based on the research.",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
)

result = crew.kickoff(inputs={"topic": "adaptive AI systems"})
```

## Setup

```bash
pip install langchain-kalibr crewai
export KALIBR_API_KEY="..."       # from dashboard.kalibr.systems
export KALIBR_TENANT_ID="..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Since CrewAI accepts any LangChain LLM, `ChatKalibr` works as a drop-in. Each agent gets its own routing goal, so Kalibr learns optimal models per agent role independently.

## Links

- [langchain-kalibr on PyPI](https://pypi.org/project/langchain-kalibr/)
- [Kalibr SDK](https://pypi.org/project/kalibr/)
- [Dashboard](https://dashboard.kalibr.systems)
- [Docs](https://kalibr.systems/docs)

Happy to answer any questions or help with integration. We also have a native CrewAI callback handler in the main `kalibr` package if you want full tracing: `pip install kalibr[crewai]`.
