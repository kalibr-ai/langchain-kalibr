"""Tests for langchain-kalibr ChatKalibr model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_kalibr import ChatKalibr


# ── Fixtures ──────────────────────────────────────────────────────────


def _mock_openai_response(content: str = "Hello!", model: str = "gpt-4o") -> SimpleNamespace:
    """Create a mock OpenAI-style response."""
    return SimpleNamespace(
        id="chatcmpl-test123",
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        kalibr_trace_id="abc123",
    )


class MockRouter:
    """Mock Kalibr Router for testing without network calls."""

    def __init__(self, goal: str, paths: list, **kwargs: Any):
        self.goal = goal
        self.paths = paths
        self.success_when = kwargs.get("success_when")
        self.exploration_rate = kwargs.get("exploration_rate")
        self.auto_register = kwargs.get("auto_register", True)
        self._last_trace_id = "mock-trace-123"
        self._last_model_id = "gpt-4o"
        self._outcome_reported = False
        self._last_report: Optional[Dict] = None

    def completion(self, messages: List[Dict], **kwargs: Any) -> SimpleNamespace:
        return _mock_openai_response()

    def report(self, success: bool, reason: Optional[str] = None,
               score: Optional[float] = None, trace_id: Optional[str] = None) -> None:
        self._last_report = {
            "success": success,
            "reason": reason,
            "score": score,
            "trace_id": trace_id,
        }
        self._outcome_reported = True


# ── Tests ─────────────────────────────────────────────────────────────


@patch.dict("os.environ", {
    "KALIBR_API_KEY": "test-key",
    "KALIBR_TENANT_ID": "test-tenant",
})
@patch("kalibr.Router", MockRouter)
class TestChatKalibr:
    """Test ChatKalibr LangChain integration."""

    def test_basic_invoke(self):
        """Test basic invoke with string messages."""
        llm = ChatKalibr(goal="test", paths=["gpt-4o"])
        result = llm.invoke("Hello!")

        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"

    def test_invoke_with_message_objects(self):
        """Test invoke with LangChain message objects."""
        llm = ChatKalibr(goal="test", paths=["gpt-4o"])
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hi there!"),
        ]
        result = llm.invoke(messages)

        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"

    def test_llm_type(self):
        """Test _llm_type returns 'kalibr'."""
        llm = ChatKalibr(goal="test", paths=["gpt-4o"])
        assert llm._llm_type == "kalibr"

    def test_identifying_params(self):
        """Test identifying params include goal and paths."""
        llm = ChatKalibr(goal="summarize", paths=["gpt-4o", "claude-sonnet-4-20250514"])
        params = llm._identifying_params

        assert params["goal"] == "summarize"
        assert params["paths"] == ["gpt-4o", "claude-sonnet-4-20250514"]

    def test_report_outcome(self):
        """Test outcome reporting."""
        llm = ChatKalibr(goal="test", paths=["gpt-4o"])
        llm.invoke("Hello!")
        llm.report(success=True)

        assert llm._router._last_report["success"] is True

    def test_report_with_reason(self):
        """Test outcome reporting with failure reason."""
        llm = ChatKalibr(goal="test", paths=["gpt-4o"])
        llm.invoke("Hello!")
        llm.report(success=False, reason="Output was empty")

        assert llm._router._last_report["success"] is False
        assert llm._router._last_report["reason"] == "Output was empty"

    def test_last_trace_id(self):
        """Test last_trace_id property."""
        llm = ChatKalibr(goal="test", paths=["gpt-4o"])
        llm.invoke("Hello!")

        assert llm.last_trace_id == "mock-trace-123"

    def test_last_model_id(self):
        """Test last_model_id property."""
        llm = ChatKalibr(goal="test", paths=["gpt-4o"])
        llm.invoke("Hello!")

        assert llm.last_model_id == "gpt-4o"

    def test_multiple_paths(self):
        """Test initialization with multiple paths."""
        llm = ChatKalibr(
            goal="extract",
            paths=["gpt-4o", "claude-sonnet-4-20250514", "gpt-4o-mini"],
        )
        assert len(llm.paths) == 3

    def test_dict_paths(self):
        """Test initialization with dict-style paths."""
        llm = ChatKalibr(
            goal="research",
            paths=[
                {"model": "gpt-4o", "tools": ["web_search"]},
                {"model": "claude-sonnet-4-20250514"},
            ],
        )
        assert len(llm.paths) == 2

    def test_success_when_callback(self):
        """Test success_when is passed to router."""
        checker = lambda out: "@" in out  # noqa: E731
        llm = ChatKalibr(
            goal="extract_email",
            paths=["gpt-4o"],
            success_when=checker,
        )
        assert llm._router.success_when is checker

    def test_response_metadata(self):
        """Test that response includes Kalibr metadata."""
        llm = ChatKalibr(goal="test", paths=["gpt-4o"])
        result = llm.invoke("Hello!")

        assert "kalibr_trace_id" in result.response_metadata
        assert result.response_metadata["model"] == "gpt-4o"

    def test_chain_compatible(self):
        """Test that ChatKalibr works in a LCEL chain."""
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatKalibr(goal="test", paths=["gpt-4o"])
        chain = llm | StrOutputParser()
        result = chain.invoke("Hello!")

        assert isinstance(result, str)
        assert result == "Hello!"
