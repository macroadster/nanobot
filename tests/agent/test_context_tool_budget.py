"""Tool schemas shrink to the active model's input budget."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.runner_helpers import make_run_spec
from nanobot.agent.context_governance import (
    ContextGovernanceConfig,
    ContextGovernor,
)
from nanobot.agent.runner import AgentRunner
from nanobot.providers.base import LLMProvider, LLMResponse, LLMUsage


def _config(window: int, max_tokens: int = 100) -> ContextGovernanceConfig:
    return ContextGovernanceConfig(
        provider=MagicMock(spec=LLMProvider),
        model="qwen3.8-iq2s:32k",
        tools=MagicMock(),
        workspace=None,
        session_key="websocket:test",
        max_tool_result_chars=1000,
        context_window_tokens=window,
        max_tokens=max_tokens,
    )


def _schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def _install_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    def estimate(_provider, _model, messages, tools):
        del messages
        cost = 100
        for tool in tools or []:
            name = tool["function"]["name"]
            cost += 5000 if name == "mcp_big" else 100
        return cost, "test-counter"

    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain",
        estimate,
    )


def test_small_window_keeps_builtins_and_small_mcp_tools(monkeypatch):
    _install_counter(monkeypatch)
    # 1624 - 100 - 1024 = 500. Full set costs 5300; builtin + mcp_small costs 300.
    tools = [_schema("mcp_big"), _schema("read_file"), _schema("mcp_small")]

    fitted = ContextGovernor().fit_tool_definitions(
        _config(1_624),
        [{"role": "system", "content": "identity"}, {"role": "user", "content": "hello"}],
        tools,
    )

    assert [tool["function"]["name"] for tool in fitted or []] == ["read_file", "mcp_small"]


def test_large_window_keeps_every_tool_schema(monkeypatch):
    _install_counter(monkeypatch)
    tools = [_schema("read_file"), _schema("mcp_big")]

    fitted = ContextGovernor().fit_tool_definitions(
        _config(200_000, max_tokens=8192),
        [{"role": "user", "content": "hello"}],
        tools,
    )

    assert fitted is tools


@pytest.mark.asyncio
async def test_runner_sends_only_tools_that_fit(monkeypatch):
    _install_counter(monkeypatch)
    provider = MagicMock(spec=LLMProvider)
    captured: dict = {}

    async def chat_stream_with_retry(**kwargs):
        captured.update(kwargs)
        return LLMResponse(content="pong", usage=LLMUsage.reported(input_tokens=10, output_tokens=1))

    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = [
        _schema("mcp_big"),
        _schema("read_file"),
        _schema("mcp_small"),
    ]

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[
            {"role": "system", "content": "identity"},
            {"role": "user", "content": "hello"},
        ],
        tools=tools,
        model="qwen3.8-iq2s:32k",
        context_window_tokens=1_624,
        max_tokens=100,
        max_iterations=1,
        max_tool_result_chars=1000,
    ))

    assert result.final_content == "pong"
    assert [tool["function"]["name"] for tool in captured["tools"]] == ["read_file", "mcp_small"]
