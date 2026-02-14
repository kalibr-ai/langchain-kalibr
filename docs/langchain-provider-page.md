# Kalibr

>[Kalibr](https://kalibr.systems) provides adaptive model routing for AI agents.
> Instead of hardcoding a single LLM, Kalibr learns which models work best for
> each task and routes automatically using Thompson Sampling.

## Installation and Setup

```bash
pip install langchain-kalibr
```

Get your API key and Tenant ID from
[dashboard.kalibr.systems/settings](https://dashboard.kalibr.systems/settings):

```bash
export KALIBR_API_KEY="your-api-key"
export KALIBR_TENANT_ID="your-tenant-id"
```

## Chat model

`ChatKalibr` provides a drop-in LangChain chat model that routes between
multiple LLM providers (OpenAI, Anthropic, Google, etc.) and learns from
outcomes to improve routing over time.

```python
from langchain_kalibr import ChatKalibr

llm = ChatKalibr(
    goal="summarize",
    paths=["gpt-4o", "claude-sonnet-4-20250514"],
)
```

See a [usage example](/docs/integrations/chat/kalibr).

For more detail, see the [langchain-kalibr API reference](https://pypi.org/project/langchain-kalibr/).
