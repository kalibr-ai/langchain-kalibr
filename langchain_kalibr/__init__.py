"""Kalibr LangChain integration — autonomous execution path routing for AI agents.

Kalibr routes your agents to the optimal execution path (model + tool + parameters)
to prevent failures, degradations, and cost spikes before they impact users.

Install:
    pip install langchain-kalibr

Usage:
    from langchain_kalibr import ChatKalibr

    llm = ChatKalibr(
        goal="summarize",
        paths=["gpt-4o", "claude-sonnet-4-20250514"],
    )

    chain = prompt | llm | parser
    result = chain.invoke({"text": "..."})

    llm.report(success=True)

Environment Variables:
    KALIBR_API_KEY: API key from dashboard.kalibr.systems
    KALIBR_TENANT_ID: Tenant ID from dashboard.kalibr.systems
    OPENAI_API_KEY: For routing to OpenAI models
    ANTHROPIC_API_KEY: For routing to Anthropic models
    GOOGLE_API_KEY: For routing to Google models
"""

__version__ = "0.1.0"

from langchain_kalibr.chat_models import ChatKalibr

__all__ = [
    "ChatKalibr",
    "__version__",
]
