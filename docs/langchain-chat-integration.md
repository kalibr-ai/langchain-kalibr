# ChatKalibr

This notebook provides a quick overview for getting started with Kalibr
[chat models](/docs/integrations/chat/). For detailed documentation of all
`ChatKalibr` features and configurations, head to the
[API reference](https://pypi.org/project/langchain-kalibr/).

## Overview

Kalibr provides adaptive model routing for AI agents. Instead of hardcoding
a single LLM provider, `ChatKalibr` lets you define multiple models and
learns which one works best for each task — routing traffic accordingly
using Thompson Sampling.

### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
|:------|:--------|:-----:|:------------:|:----------:|:-----------------:|:-------------:|
| `ChatKalibr` | `langchain-kalibr` | ❌ | ❌ | ❌ | ![PyPI Downloads](https://img.shields.io/pypi/dm/langchain-kalibr) | ![PyPI Version](https://img.shields.io/pypi/v/langchain-kalibr) |

### Model features

| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

## Setup

### Credentials

Get your API key and Tenant ID from
[dashboard.kalibr.systems/settings](https://dashboard.kalibr.systems/settings):

```python
import os

os.environ["KALIBR_API_KEY"] = "your-api-key"
os.environ["KALIBR_TENANT_ID"] = "your-tenant-id"

# Provider keys for models you want to route between:
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

### Installation

```bash
pip install -qU langchain-kalibr
```

## Instantiation

```python
from langchain_kalibr import ChatKalibr

llm = ChatKalibr(
    goal="summarize",
    paths=["gpt-4o", "claude-sonnet-4-20250514"],
)
```

## Invocation

```python
messages = [
    ("system", "You are a helpful assistant that summarizes text concisely."),
    ("human", "Explain adaptive model routing in one paragraph."),
]

response = llm.invoke(messages)
print(response.content)
```

## Outcome Reporting

What makes Kalibr different is the feedback loop. After each call,
report whether the task succeeded:

```python
response = llm.invoke("Extract the email: Contact hello@example.com for details.")
llm.report(success="@" in response.content)
```

Or use auto-reporting:

```python
llm = ChatKalibr(
    goal="extract_email",
    paths=["gpt-4o", "claude-sonnet-4-20250514"],
    success_when=lambda output: "@" in output,
)

# Outcome reported automatically
response = llm.invoke("Extract the email: Contact hello@example.com for details.")
```

## Chaining

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "Thompson Sampling"})
```

## Advanced: Path Configuration

Route between different model configurations:

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

## API reference

For detailed documentation, see the
[langchain-kalibr PyPI page](https://pypi.org/project/langchain-kalibr/)
and [Kalibr documentation](https://kalibr.systems/docs).
