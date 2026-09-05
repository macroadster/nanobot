"""Background SSE stream monitors that wake a subagent on each event."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.context import (
    RequestContext,
    bind_request_context,
    reset_request_context,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage
from nanobot.security.network import (
    PinnedDNSAsyncTransport,
    httpx_env_proxy_mounts,
    validate_url_target,
)
from nanobot.security.workspace_access import (
    WorkspaceScope,
    bind_workspace_scope,
    reset_workspace_scope,
)
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


MAX_CONCURRENT_MONITORS = 3
MAX_HISTORY_MESSAGES = 20
MAX_EVENT_ITERATIONS = 20
_RECONNECT_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)
_SKIP_EVENT_TYPES = frozenset({"typing", "ping", "keepalive", "heartbeat"})
_EXTRA_PARENT_TOOLS = frozenset({"message"})


@dataclass(slots=True)
class SSEEvent:
    """One dispatched Server-Sent Event."""

    event: str = "message"
    data: str = ""
    id: str | None = None


@dataclass(slots=True)
class StreamMonitorStatus:
    """Observable status of one live stream monitor."""

    monitor_id: str
    label: str
    url: str
    task: str
    started_at: float
    phase: str = "connecting"  # connecting | listening | handling | reconnecting | stopped | error
    events_seen: int = 0
    events_dispatched: int = 0
    last_event_at: float | None = None
    error: str | None = None
    agent_id: str | None = None
    room_id: str | None = None


class SSEParser:
    """Incremental parser for the text/event-stream line protocol."""

    def __init__(self) -> None:
        self._event = "message"
        self._event_id: str | None = None
        self._data: list[str] = []

    def push(self, line: str) -> SSEEvent | None:
        if line.endswith("\r"):
            line = line[:-1]
        if line == "":
            return self._flush()
        if line.startswith(":"):
            return None
        field, sep, value = line.partition(":")
        if sep:
            if value.startswith(" "):
                value = value[1:]
        else:
            value = ""
        if field == "event":
            self._event = value or "message"
        elif field == "data":
            self._data.append(value)
        elif field == "id":
            if "\x00" not in value:
                self._event_id = value or None
        return None

    def _flush(self) -> SSEEvent | None:
        if not self._data:
            self._event = "message"
            return None
        event = SSEEvent(
            event=self._event or "message",
            data="\n".join(self._data),
            id=self._event_id,
        )
        self._event = "message"
        self._data = []
        return event


def coerce_stream_url(raw: str) -> str:
    """Accept a URL, or a chat_stream-style JSON payload that contains ``stream_url``."""
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(payload, dict):
            stream_url = payload.get("stream_url") or payload.get("url")
            if isinstance(stream_url, str) and stream_url.strip():
                return stream_url.strip()
    return text


def resolve_stream_url(url: str, room_id: str | None, agent_id: str | None) -> str:
    """Append room/agent query params when the URL does not already carry them."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if room_id and "room" not in query:
        query["room"] = room_id
    if agent_id and "agent" not in query:
        query["agent"] = agent_id
    return urlunparse(parsed._replace(query=urlencode(query)))


def headers_for_stream(
    url: str,
    mcp_servers: dict[str, Any] | None,
    mcp_server: str | None,
) -> dict[str, str]:
    """Copy Authorization headers from a matching MCP server config."""
    servers = mcp_servers or {}
    if mcp_server:
        cfg = servers.get(mcp_server)
        return _config_headers(cfg)
    target = urlparse(url).netloc
    if not target:
        return {}
    for cfg in servers.values():
        cfg_url = getattr(cfg, "url", "") or ""
        if cfg_url and urlparse(cfg_url).netloc == target:
            return _config_headers(cfg)
    return {}


def decode_event_data(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def should_dispatch_event(payload: Any, agent_id: str | None) -> bool:
    """Skip keepalives, typing indicators, and the monitor's own echoes."""
    if payload in ("", None):
        return False
    if isinstance(payload, str):
        return bool(payload.strip())
    if not isinstance(payload, dict):
        return True
    event_type = str(
        payload.get("type") or payload.get("event") or "message"
    ).strip().lower()
    if event_type in _SKIP_EVENT_TYPES:
        return False
    sender = payload.get("agent_id") or payload.get("actor") or payload.get("sender")
    if agent_id and sender is not None and str(sender) == agent_id:
        return False
    return True


def format_event_for_prompt(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(payload)


def _config_headers(cfg: Any) -> dict[str, str]:
    if cfg is None:
        return {}
    raw = getattr(cfg, "headers", None)
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _redact_url(url: str) -> str:
    try:
        parts = urlparse(url)
        hostname = parts.hostname or ""
        netloc = f"[{hostname}]" if ":" in hostname else hostname
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        path = "/..." if parts.path and parts.path != "/" else parts.path
        return urlunparse((parts.scheme, netloc, path, "", "", ""))
    except Exception:
        return "<redacted-url>"


def _pinned_transport_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {"transport": PinnedDNSAsyncTransport()}
    mounts = httpx_env_proxy_mounts()
    if mounts:
        kwargs["mounts"] = mounts
    return kwargs


async def _validate_request_url(request: httpx.Request) -> None:
    ok, error = validate_url_target(str(request.url))
    if not ok:
        raise httpx.RequestError(
            f"Blocked unsafe stream URL {_redact_url(str(request.url))} ({error})",
            request=request,
        )


class StreamMonitorManager:
    """Owns long-lived SSE subscriptions and runs a subagent turn per event."""

    def __init__(
        self,
        *,
        workspace: Path,
        bus: Any,
        subagents: SubagentManager,
        parent_registry: ToolRegistry | None = None,
        mcp_servers: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
        max_monitors: int = MAX_CONCURRENT_MONITORS,
        llm_wall_timeout_for_session: Callable[[str | None], float | None] | None = None,
        runner: AgentRunner | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.workspace = workspace
        self.bus = bus
        self._subagents = subagents
        self._parent_registry = parent_registry
        self._mcp_servers = mcp_servers or {}
        self.max_monitors = max_monitors
        self._llm_wall_timeout_for_session = llm_wall_timeout_for_session
        self.runner = runner or AgentRunner()
        self._client_factory = client_factory
        self._monitors: dict[str, StreamMonitorStatus] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._histories: dict[str, list[dict[str, Any]]] = {}
        self._origins: dict[str, dict[str, str]] = {}
        self._runtimes: dict[str, LLMRuntime] = {}
        self._scopes: dict[str, WorkspaceScope | None] = {}

    @property
    def _task_statuses(self) -> dict[str, StreamMonitorStatus]:
        """Alias so ``my(check="stream_monitors")`` can format live status."""
        return self._monitors

    def _resolved_mcp_servers(self) -> dict[str, Any]:
        servers = self._mcp_servers
        if callable(servers):
            return servers() or {}
        return servers or {}

    def get_running_count(self) -> int:
        return sum(1 for task in self._tasks.values() if not task.done())

    def list_statuses(self) -> list[StreamMonitorStatus]:
        return list(self._monitors.values())

    def format_list(self) -> str:
        rows = self.list_statuses()
        if not rows:
            return "No stream monitors running."
        lines = [f"{len(rows)} stream monitor(s):"]
        now = time.monotonic()
        for status in rows:
            last = (
                f"{now - status.last_event_at:.0f}s ago"
                if status.last_event_at is not None
                else "none"
            )
            extra = ""
            if status.room_id:
                extra += f" room={status.room_id}"
            if status.agent_id:
                extra += f" agent={status.agent_id}"
            lines.append(
                f"- {status.monitor_id} [{status.label}] {status.phase} "
                f"events={status.events_dispatched}/{status.events_seen} last={last}"
                f"{extra}"
            )
            if status.error:
                lines.append(f"  error: {status.error}")
        return "\n".join(lines)

    async def start(
        self,
        *,
        url: str,
        task: str,
        runtime: LLMRuntime,
        label: str | None = None,
        agent_id: str | None = None,
        room_id: str | None = None,
        mcp_server: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        origin_message_id: str | None = None,
        workspace_scope: WorkspaceScope | None = None,
    ) -> str:
        if self.get_running_count() >= self.max_monitors:
            return (
                f"Cannot start stream monitor: limit reached "
                f"({self.get_running_count()}/{self.max_monitors}). "
                "Stop an existing monitor first."
            )
        resolved = resolve_stream_url(coerce_stream_url(url), room_id, agent_id)
        if not resolved:
            return "Error: url is required to start a stream monitor."
        ok, error = validate_url_target(resolved)
        if not ok:
            return f"Error: blocked unsafe stream URL ({error})"

        monitor_id = str(uuid.uuid4())[:8]
        display = label or (room_id or resolved)[:40]
        status = StreamMonitorStatus(
            monitor_id=monitor_id,
            label=display,
            url=resolved,
            task=task,
            started_at=time.monotonic(),
            agent_id=agent_id,
            room_id=room_id,
        )
        stop = asyncio.Event()
        self._monitors[monitor_id] = status
        self._stop_events[monitor_id] = stop
        self._histories[monitor_id] = []
        self._runtimes[monitor_id] = runtime
        self._scopes[monitor_id] = workspace_scope
        self._origins[monitor_id] = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
            "session_key": session_key or "",
            "message_id": origin_message_id or "",
        }
        headers = headers_for_stream(resolved, self._resolved_mcp_servers(), mcp_server)
        bg = asyncio.create_task(
            self._run_monitor(monitor_id, headers),
            name=f"stream-monitor:{monitor_id}",
        )
        self._tasks[monitor_id] = bg
        bg.add_done_callback(lambda _task, mid=monitor_id: self._forget(mid))
        logger.info("Started stream monitor [{}]: {}", monitor_id, _redact_url(resolved))
        return (
            f"Stream monitor [{display}] started (id: {monitor_id}). "
            f"Listening on {_redact_url(resolved)}. "
            "A subagent will handle each event and can reply via MCP tools "
            f"(for example chat_send). Stop with monitor(action=\"stop\", monitor_id=\"{monitor_id}\")."
        )

    async def stop(self, monitor_id: str) -> str:
        status = self._monitors.get(monitor_id)
        if status is None:
            return f"Error: no stream monitor with id '{monitor_id}'."
        await self._stop_one(monitor_id, reason="stopped")
        return f"Stopped stream monitor [{status.label}] ({monitor_id})."

    async def close(self) -> None:
        ids = list(self._tasks)
        for monitor_id in ids:
            await self._stop_one(monitor_id, reason="shutdown", notify=False)

    async def _stop_one(
        self,
        monitor_id: str,
        *,
        reason: str,
        notify: bool = False,
    ) -> None:
        stop = self._stop_events.get(monitor_id)
        if stop is not None:
            stop.set()
        task = self._tasks.get(monitor_id)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        status = self._monitors.get(monitor_id)
        if status is not None:
            status.phase = "stopped" if reason == "stopped" else status.phase
            if reason not in {"stopped", "shutdown"}:
                status.error = reason
                status.phase = "error"
        if notify and status is not None:
            await self._announce(status, reason)
        self._forget(monitor_id)

    def _forget(self, monitor_id: str) -> None:
        self._tasks.pop(monitor_id, None)
        self._stop_events.pop(monitor_id, None)
        self._histories.pop(monitor_id, None)
        self._origins.pop(monitor_id, None)
        self._runtimes.pop(monitor_id, None)
        self._scopes.pop(monitor_id, None)
        self._monitors.pop(monitor_id, None)

    async def _run_monitor(self, monitor_id: str, headers: dict[str, str]) -> None:
        status = self._monitors[monitor_id]
        stop = self._stop_events[monitor_id]
        backoff_idx = 0
        last_event_id: str | None = None
        try:
            while not stop.is_set():
                try:
                    async for event in self._iter_sse(status.url, headers, last_event_id, stop):
                        backoff_idx = 0
                        status.phase = "listening"
                        status.events_seen += 1
                        status.last_event_at = time.monotonic()
                        if event.id:
                            last_event_id = event.id
                        payload = decode_event_data(event.data)
                        if event.event and event.event != "message" and isinstance(payload, dict):
                            payload.setdefault("event", event.event)
                        if not should_dispatch_event(payload, status.agent_id):
                            continue
                        status.events_dispatched += 1
                        status.phase = "handling"
                        try:
                            await self._handle_event(monitor_id, payload)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "Stream monitor [{}] failed handling an event",
                                monitor_id,
                            )
                        status.phase = "listening"
                    if stop.is_set():
                        break
                    raise httpx.TransportError("stream closed")
                except asyncio.CancelledError:
                    raise
                except httpx.HTTPStatusError as exc:
                    code = exc.response.status_code if exc.response is not None else 0
                    if code in {401, 403, 404}:
                        reason = f"stream returned HTTP {code}"
                        status.phase = "error"
                        status.error = reason
                        await self._announce(status, reason)
                        return
                    logger.warning(
                        "Stream monitor [{}] HTTP error on {}: {}",
                        monitor_id,
                        _redact_url(status.url),
                        exc,
                    )
                except Exception as exc:
                    logger.warning(
                        "Stream monitor [{}] disconnected from {}: {}",
                        monitor_id,
                        _redact_url(status.url),
                        exc,
                    )
                if stop.is_set():
                    break
                delay = _RECONNECT_BACKOFF_S[min(backoff_idx, len(_RECONNECT_BACKOFF_S) - 1)]
                backoff_idx += 1
                status.phase = "reconnecting"
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            status.phase = "stopped"
            raise
        finally:
            if status.phase not in {"error", "stopped"}:
                status.phase = "stopped"

    async def _iter_sse(
        self,
        url: str,
        headers: dict[str, str],
        last_event_id: str | None,
        stop: asyncio.Event,
    ):
        request_headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            **headers,
        }
        if last_event_id:
            request_headers["Last-Event-ID"] = last_event_id
        timeout = httpx.Timeout(None, connect=10.0)
        client = self._client_factory() if self._client_factory is not None else None
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(
                event_hooks={"request": [_validate_request_url]},
                follow_redirects=True,
                timeout=timeout,
                **_pinned_transport_kwargs(),
            )
        try:
            parser = SSEParser()
            async with client.stream("GET", url, headers=request_headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if stop.is_set():
                        return
                    event = parser.push(line)
                    if event is not None:
                        yield event
        finally:
            if owns_client:
                await client.aclose()

    async def _handle_event(self, monitor_id: str, payload: Any) -> None:
        status = self._monitors.get(monitor_id)
        runtime = self._runtimes.get(monitor_id)
        origin = self._origins.get(monitor_id)
        if status is None or runtime is None or origin is None:
            return
        history = self._histories.setdefault(monitor_id, [])
        scope = self._scopes.get(monitor_id)
        workspace = scope.project_path if scope is not None else self.workspace
        user_content = render_template(
            "agent/stream_monitor_event.md",
            event=format_event_for_prompt(payload),
        )
        system_prompt = render_template(
            "agent/stream_monitor_system.md",
            task=status.task,
            agent_id=status.agent_id or "",
            room_id=status.room_id or "",
            workspace=str(workspace),
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_content},
        ]
        tools = self._event_tools(self._scopes.get(monitor_id))
        sess_key = origin.get("session_key") or None
        llm_timeout = (
            self._llm_wall_timeout_for_session(sess_key)
            if self._llm_wall_timeout_for_session
            else None
        )
        request_token = bind_request_context(RequestContext(
            channel=origin["channel"],
            chat_id=origin["chat_id"],
            message_id=origin.get("message_id") or None,
            session_key=sess_key,
            runtime=runtime,
        ))
        scope = self._scopes.get(monitor_id)
        scope_token = bind_workspace_scope(scope) if scope is not None else None
        try:
            result = await self.runner.run(AgentRunSpec(
                initial_messages=messages,
                tools=tools,
                runtime=runtime,
                max_iterations=min(self._subagents.max_iterations, MAX_EVENT_ITERATIONS),
                max_tool_result_chars=self._subagents.max_tool_result_chars,
                max_iterations_message="Stop. Reply only if the latest event still needs a response.",
                finalize_on_max_iterations=True,
                error_message=None,
                session_key=sess_key,
                workspace=self.workspace,
                llm_timeout_s=llm_timeout,
            ))
        finally:
            if scope_token is not None:
                reset_workspace_scope(scope_token)
            reset_request_context(request_token)

        reply = (result.final_content or "").strip()
        history.append({"role": "user", "content": user_content})
        history.append({
            "role": "assistant",
            "content": reply or "(no reply)",
        })
        overflow = len(history) - MAX_HISTORY_MESSAGES
        if overflow > 0:
            del history[:overflow]

    def _event_tools(self, workspace_scope: WorkspaceScope | None) -> ToolRegistry:
        cfg = None
        if workspace_scope is not None:
            cfg = self._subagents._subagent_tools_config()
            cfg.restrict_to_workspace = workspace_scope.restrict_to_workspace
        tools = self._subagents._build_tools(tools_config=cfg)
        registry = self._parent_registry
        if registry is None:
            return tools
        for name in registry.tool_names:
            if not (name.startswith("mcp_") or name in _EXTRA_PARENT_TOOLS):
                continue
            if tools.has(name):
                continue
            tool = registry.get(name)
            if tool is not None:
                tools.register(tool)
        return tools

    async def _announce(self, status: StreamMonitorStatus, reason: str) -> None:
        origin = self._origins.get(status.monitor_id)
        if origin is None or self.bus is None:
            return
        content = render_template(
            "agent/stream_monitor_announce.md",
            label=status.label,
            reason=reason,
            monitor_id=status.monitor_id,
        )
        override = origin.get("session_key") or f"{origin['channel']}:{origin['chat_id']}"
        msg = InboundMessage(
            channel="system",
            sender_id="stream_monitor",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=content,
            session_key_override=override,
            metadata={
                "injected_event": "stream_monitor_stopped",
                "stream_monitor_id": status.monitor_id,
            },
        )
        await self.bus.publish_inbound(msg)
