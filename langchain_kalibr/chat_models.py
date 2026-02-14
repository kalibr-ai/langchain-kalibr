"""Kalibr chat model for LangChain.

Adaptive model routing that learns which models work best for your tasks.

Example:
    .. code-block:: python

        from langchain_kalibr import ChatKalibr

        llm = ChatKalibr(
            goal="summarize",
            paths=["gpt-4o", "claude-sonnet-4-20250514"],
        )

        # Use in any LangChain chain
        response = llm.invoke("Summarize the key points of this document...")

        # Report outcomes to improve routing
        llm.report(success=True)
"""

from __future__ import annotations

import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, model_validator

logger = logging.getLogger(__name__)


def _message_to_openai(message: BaseMessage) -> Dict[str, str]:
    """Convert a LangChain message to OpenAI dict format."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content or ""}
    elif isinstance(message, ToolMessage):
        return {"role": "tool", "content": message.content}
    else:
        return {"role": "user", "content": message.content}


class ChatKalibr(BaseChatModel):
    """Kalibr adaptive routing chat model for LangChain.

    Routes requests to the best-performing model (OpenAI, Anthropic, Google, etc.)
    based on learned outcomes. Uses Thompson Sampling to balance exploration vs
    exploitation — automatically routing more traffic to models that work.

    Setup:
        Install ``langchain-kalibr`` and set environment variables:

        .. code-block:: bash

            pip install langchain-kalibr
            export KALIBR_API_KEY="your-api-key"       # from dashboard.kalibr.systems
            export KALIBR_TENANT_ID="your-tenant-id"   # from dashboard.kalibr.systems
            export OPENAI_API_KEY="sk-..."             # for OpenAI models
            export ANTHROPIC_API_KEY="sk-ant-..."      # for Anthropic models

    Key init args:
        goal: str
            Name of the task/goal for routing (e.g., "summarize", "extract_email").
        paths: List[str | dict]
            Models or model configs to route between.
        success_when: Optional callable
            Auto-evaluate success from output text.

    Instantiate:
        .. code-block:: python

            from langchain_kalibr import ChatKalibr

            llm = ChatKalibr(
                goal="summarize",
                paths=["gpt-4o", "claude-sonnet-4-20250514"],
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "Summarize this document..."),
            ]
            response = llm.invoke(messages)

    Use in a chain:
        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                ("human", "{input}"),
            ])

            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"input": "What is adaptive routing?"})

    Report outcomes (teaches Kalibr what works):
        .. code-block:: python

            response = llm.invoke("Extract the company name from: Hi, I'm Sarah from Stripe.")
            llm.report(success="Stripe" in response.content)

    """  # noqa: E501

    # ── Required fields ──────────────────────────────────────────────
    goal: str = Field(
        description="Name of the task/goal for routing (e.g., 'summarize', 'extract_email')."
    )
    paths: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=lambda: ["gpt-4o"],
        description=(
            "Models or model configs to route between. "
            "Examples: ['gpt-4o', 'claude-sonnet-4-20250514'] or "
            "[{'model': 'gpt-4o', 'tools': ['web_search']}]"
        ),
    )

    # ── Optional config ──────────────────────────────────────────────
    kalibr_api_key: Optional[str] = Field(default=None, alias="api_key")
    kalibr_tenant_id: Optional[str] = Field(default=None, alias="tenant_id")
    success_when: Optional[Callable[[str], bool]] = Field(
        default=None,
        description=(
            "Optional function to auto-evaluate success from LLM output text. "
            "Example: success_when=lambda out: '@' in out"
        ),
    )
    exploration_rate: Optional[float] = Field(
        default=None,
        description="Override exploration rate (0.0-1.0). Default: Kalibr decides.",
    )
    auto_register: bool = Field(
        default=True,
        description="If True, register paths with Kalibr on init.",
    )

    # ── Internal state (not serialized) ──────────────────────────────
    _router: Any = None

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    @model_validator(mode="after")
    def _init_router(self) -> "ChatKalibr":
        """Initialize the Kalibr Router after all fields are set."""
        # Set env vars if provided explicitly
        if self.kalibr_api_key:
            os.environ.setdefault("KALIBR_API_KEY", self.kalibr_api_key)
        if self.kalibr_tenant_id:
            os.environ.setdefault("KALIBR_TENANT_ID", self.kalibr_tenant_id)

        try:
            from kalibr import Router

            self._router = Router(
                goal=self.goal,
                paths=self.paths,
                success_when=self.success_when,
                exploration_rate=self.exploration_rate,
                auto_register=self.auto_register,
            )
        except ImportError:
            raise ImportError(
                "Could not import kalibr. "
                "Please install it with: pip install kalibr"
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to initialize Kalibr Router: {e}\n"
                "Make sure KALIBR_API_KEY and KALIBR_TENANT_ID are set.\n"
                "Get credentials at: https://dashboard.kalibr.systems/settings"
            )
        return self

    # ── LangChain interface ──────────────────────────────────────────

    @property
    def _llm_type(self) -> str:
        return "kalibr"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "goal": self.goal,
            "paths": self.paths,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Route and generate a response using Kalibr."""
        openai_messages = [_message_to_openai(m) for m in messages]

        if stop:
            kwargs["stop"] = stop

        response = self._router.completion(messages=openai_messages, **kwargs)

        content = response.choices[0].message.content or ""
        finish_reason = getattr(response.choices[0], "finish_reason", "stop")

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        generation = ChatGeneration(
            message=AIMessage(
                content=content,
                response_metadata={
                    "model": getattr(response, "model", ""),
                    "finish_reason": finish_reason,
                    "kalibr_trace_id": getattr(response, "kalibr_trace_id", None),
                },
            ),
            generation_info={
                "model": getattr(response, "model", ""),
                "finish_reason": finish_reason,
            },
        )

        return ChatResult(
            generations=[generation],
            llm_output={
                "model": getattr(response, "model", ""),
                "usage": usage,
                "kalibr_trace_id": getattr(response, "kalibr_trace_id", None),
            },
        )

    # ── Outcome reporting ────────────────────────────────────────────

    def report(
        self,
        success: bool,
        reason: Optional[str] = None,
        score: Optional[float] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """Report outcome for the last completion to improve routing.

        This teaches Kalibr which models work best for your goal.
        Call after each invoke() to close the feedback loop.

        Args:
            success: Whether the task succeeded.
            reason: Optional failure reason.
            score: Optional quality score (0.0-1.0).
            trace_id: Optional explicit trace ID.
        """
        if self._router is None:
            raise RuntimeError("Router not initialized. Check your Kalibr credentials.")
        self._router.report(success=success, reason=reason, score=score, trace_id=trace_id)

    @property
    def last_trace_id(self) -> Optional[str]:
        """Get the trace_id from the last completion (for explicit outcome reporting)."""
        if self._router:
            return self._router._last_trace_id
        return None

    @property
    def last_model_id(self) -> Optional[str]:
        """Get which model was selected for the last completion."""
        if self._router:
            return self._router._last_model_id
        return None
