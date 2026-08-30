"""Tests for SSE parsing and StreamMonitorManager."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nanobot.agent.stream_monitor import (
    MAX_CONCURRENT_MONITORS,
    SSEParser,
    StreamMonitorManager,
    coerce_stream_url,
    decode_event_data,
    format_event_for_prompt,
    headers_for_stream,
    resolve_stream_url,
    should_dispatch_event,
)
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.monitor import MonitorTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults, MCPServerConfig
from nanobot.providers.base import GenerationSettings
from nanobot.utils.llm_runtime import LLMRuntime


def _runtime(provider: MagicMock | None = None) -> LLMRuntime:
    provider = provider or MagicMock()
    provider.generation = GenerationSettings(temperature=0.1, max_tokens=4096)
    return LLMRuntime.capture(provider, "test-model", context_window_tokens=128_000)


def _subagents(tmp_path: Path) -> MagicMock:
    from nanobot.agent.subagent import SubagentManager

    manager = MagicMock(spec=SubagentManager)
    manager.max_iterations = 40
    manager.max_tool_result_chars = AgentDefaults().max_tool_result_chars
    manager._subagent_tools_config.return_value = MagicMock()
    manager._build_tools.return_value = ToolRegistry()
    return manager


# --- helpers ---


def test_sse_parser_joins_multiline_data():
    parser = SSEParser()
    assert parser.push("event: chat") is None
    assert parser.push("data: {\"text\":") is None
    assert parser.push("data: \"hi\"}") is None
    event = parser.push("")
    assert event is not None
    assert event.event == "chat"
    assert event.data == '{"text":\n"hi"}'


def test_sse_parser_ignores_comments_and_empty_blocks():
    parser = SSEParser()
    assert parser.push(": keepalive") is None
    assert parser.push("") is None
    assert parser.push("data: ping") is None
    event = parser.push("")
    assert event is not None
    assert event.event == "message"
    assert event.data == "ping"


def test_coerce_stream_url_extracts_json_field():
    raw = json.dumps({
        "stream_url": "https://starlight.example/mcp/chat/stream",
        "message": "Use GET with room and agent",
    })
    assert coerce_stream_url(raw) == "https://starlight.example/mcp/chat/stream"


def test_resolve_stream_url_appends_missing_query():
    url = resolve_stream_url(
        "https://starlight.example/mcp/chat/stream",
        room_id="contract_1",
        agent_id="bot",
    )
    assert "room=contract_1" in url
    assert "agent=bot" in url


def test_resolve_stream_url_keeps_existing_query():
    url = resolve_stream_url(
        "https://starlight.example/mcp/chat/stream?room=keep&agent=mine",
        room_id="other",
        agent_id="other-bot",
    )
    assert "room=keep" in url
    assert "agent=mine" in url
    assert "other" not in url


def test_headers_for_stream_match_named_and_host():
    servers = {
        "starlight": MCPServerConfig(
            url="https://starlight.example/mcp",
            headers={"Authorization": "Bearer secret"},
        )
    }
    named = headers_for_stream(
        "https://other.example/stream",
        servers,
        "starlight",
    )
    assert named == {"Authorization": "Bearer secret"}
    inferred = headers_for_stream(
        "https://starlight.example/mcp/chat/stream",
        servers,
        None,
    )
    assert inferred == {"Authorization": "Bearer secret"}


def test_should_dispatch_skips_noise_and_echo():
    assert should_dispatch_event({"type": "typing", "agent_id": "x"}, "bot") is False
    assert should_dispatch_event({"type": "message", "agent_id": "bot"}, "bot") is False
    assert should_dispatch_event({"type": "message", "agent_id": "other"}, "bot") is True
    assert should_dispatch_event("hello", "bot") is True
    assert should_dispatch_event("", "bot") is False


def test_decode_and_format_event():
    payload = decode_event_data('{"content":"hi"}')
    assert payload == {"content": "hi"}
    assert "hi" in format_event_for_prompt(payload)
    assert decode_event_data("not-json") == "not-json"


# --- manager ---


class _FakeStream:
    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://starlight.example/mcp/chat/stream")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeClient:
    def __init__(self, streams: list[_FakeStream]):
        self._streams = list(streams)
        self.closed = False
        self.calls: list[str] = []

    def stream(self, method: str, url: str, headers: dict[str, str] | None = None):
        self.calls.append(url)
        stream = self._streams.pop(0) if self._streams else _FakeStream([])
        return _FakeStreamCM(stream)

    async def aclose(self) -> None:
        self.closed = True


class _FakeStreamCM:
    def __init__(self, stream: _FakeStream):
        self._stream = stream

    async def __aenter__(self) -> _FakeStream:
        return self._stream

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


@pytest.mark.asyncio
async def test_monitor_dispatches_event_to_subagent_and_skips_echo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nanobot.agent.stream_monitor.validate_url_target",
        lambda url: (True, ""),
    )
    monkeypatch.setattr("nanobot.agent.stream_monitor._RECONNECT_BACKOFF_S", (0.01,))
    bus = MessageBus()
    bus.publish_inbound = AsyncMock()
    runner = MagicMock()
    runner.run = AsyncMock(return_value=SimpleNamespace(final_content="ok"))
    parent = ToolRegistry()

    class _ChatSend:
        name = "mcp_starlight_chat_send"
        description = "send"
        parameters = {"type": "object"}

        async def execute(self, **kwargs):
            return "sent"

    parent.register(_ChatSend())  # type: ignore[arg-type]

    streams = [
        _FakeStream([
            "data: {\"type\":\"message\",\"agent_id\":\"other\",\"content\":\"hello\"}",
            "",
            "data: {\"type\":\"message\",\"agent_id\":\"bot\",\"content\":\"echo\"}",
            "",
        ]),
        _FakeStream([], status_code=401),
    ]
    client = _FakeClient(streams)

    manager = StreamMonitorManager(
        workspace=tmp_path,
        bus=bus,
        subagents=_subagents(tmp_path),
        parent_registry=parent,
        runner=runner,
        client_factory=lambda: client,
    )
    result = await manager.start(
        url="https://starlight.example/mcp/chat/stream",
        task="Reply in the room.",
        runtime=_runtime(),
        room_id="contract_1",
        agent_id="bot",
        origin_channel="cli",
        origin_chat_id="direct",
        session_key="cli:direct",
    )
    assert "started" in result
    monitor_id = next(iter(manager._monitors))
    task = manager._tasks[monitor_id]
    await asyncio.wait_for(task, timeout=2)
    assert runner.run.await_count == 1
    spec = runner.run.await_args.args[0]
    assert spec.tools.has("mcp_starlight_chat_send")
    user_text = spec.initial_messages[-1]["content"]
    assert "hello" in user_text
    assert "echo" not in user_text
    listed = manager.format_list()
    assert "No stream monitors" in listed or monitor_id not in manager._monitors


@pytest.mark.asyncio
async def test_monitor_start_rejects_unsafe_url(tmp_path):
    manager = StreamMonitorManager(
        workspace=tmp_path,
        bus=MessageBus(),
        subagents=_subagents(tmp_path),
    )
    result = await manager.start(
        url="http://127.0.0.1/mcp/chat/stream",
        task="watch",
        runtime=_runtime(),
    )
    assert result.startswith("Error: blocked unsafe stream URL")
    assert manager.get_running_count() == 0


@pytest.mark.asyncio
async def test_monitor_limit_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nanobot.agent.stream_monitor.validate_url_target",
        lambda url: (True, ""),
    )
    client = _FakeClient([_FakeStream(["data: {\"type\":\"typing\"}", ""])])
    manager = StreamMonitorManager(
        workspace=tmp_path,
        bus=MessageBus(),
        subagents=_subagents(tmp_path),
        runner=MagicMock(run=AsyncMock(return_value=SimpleNamespace(final_content=""))),
        client_factory=lambda: client,
        max_monitors=1,
    )
    first = await manager.start(
        url="https://starlight.example/mcp/chat/stream",
        task="watch",
        runtime=_runtime(),
        label="one",
    )
    assert "started" in first
    second = await manager.start(
        url="https://starlight.example/mcp/chat/stream?room=b",
        task="watch",
        runtime=_runtime(),
        label="two",
    )
    assert "limit reached" in second
    monitor_id = next(iter(manager._monitors))
    stopped = await manager.stop(monitor_id)
    assert "Stopped" in stopped
    assert manager.get_running_count() == 0


@pytest.mark.asyncio
async def test_monitor_tool_start_stop_list(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nanobot.agent.stream_monitor.validate_url_target",
        lambda url: (True, ""),
    )
    manager = StreamMonitorManager(
        workspace=tmp_path,
        bus=MessageBus(),
        subagents=_subagents(tmp_path),
        runner=MagicMock(run=AsyncMock(return_value=SimpleNamespace(final_content=""))),
        client_factory=lambda: _FakeClient([_FakeStream([])]),
    )
    tool = MonitorTool(manager)
    listed = await tool.execute(action="list")
    assert "No stream monitors" in listed
    missing = await tool.execute(action="start", url="https://example.com/stream")
    assert "task is required" in str(missing)
    with request_context(RequestContext(
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        runtime=_runtime(),
    )):
        started = await tool.execute(
            action="start",
            url=json.dumps({"stream_url": "https://starlight.example/mcp/chat/stream"}),
            room_id="contract_9",
            agent_id="nanobot",
            task="Sit in the room and reply when asked.",
            label="room-9",
        )
    assert "started" in started
    listed = await tool.execute(action="list")
    assert "room-9" in listed
    monitor_id = next(iter(manager._monitors))
    stopped = await tool.execute(action="stop", monitor_id=monitor_id)
    assert "Stopped" in stopped


@pytest.mark.asyncio
async def test_monitor_tool_requires_runtime():
    manager = MagicMock()
    tool = MonitorTool(manager)
    result = await tool.execute(
        action="start",
        url="https://starlight.example/mcp/chat/stream",
        task="watch",
    )
    assert "active model runtime" in str(result)
    manager.start.assert_not_called()


def test_max_concurrent_default():
    assert MAX_CONCURRENT_MONITORS >= 1
