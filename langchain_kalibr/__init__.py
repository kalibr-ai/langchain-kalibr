"""LangChain integration for Kalibr — adaptive model routing for AI agents.

Kalibr learns which models work best for your tasks and routes automatically.
This package provides a LangChain-compatible chat model that plugs into any
LangChain chain, agent, or LangGraph workflow.

Install:
    pip install langchain-kalibr

Usage:
    from langchain_kalibr import ChatKalibr

    llm = ChatKalibr(
        goal="summarize",
        paths=["gpt-4o", "claude-sonnet-4-20250514"],
    )

    # Drop into any LangChain chain
    chain = prompt | llm | parser
    result = chain.invoke({"text": "..."})

    # Report outcomes to teach Kalibr what works
    llm.report(success=True)

Environment Variables:
    KALIBR_API_KEY: API key from dashboard.kalibr.systems
    KALIBR_TENANT_ID: Tenant ID from dashboard.kalibr.systems
    OPENAI_API_KEY: For routing to OpenAI models
    ANTHROPIC_API_KEY: For routing to Anthropic models
"""

__version__ = "0.1.0"

from langchain_kalibr.chat_models import ChatKalibr

__all__ = [
    "ChatKalibr",
    "__version__",
]
