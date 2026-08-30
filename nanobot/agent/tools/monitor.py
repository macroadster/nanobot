"""Tool for starting and stopping SSE stream monitors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema
from nanobot.security.workspace_access import current_workspace_scope

if TYPE_CHECKING:
    from nanobot.agent.stream_monitor import StreamMonitorManager


@tool_parameters(
    tool_parameters_schema(
        action=StringSchema(
            "Action to perform",
            enum=["start", "stop", "list"],
        ),
        url=StringSchema(
            "REQUIRED when action='start'. SSE or Streamable HTTP URL to watch. "
            "You may pass the JSON object returned by an MCP chat_stream / "
            "events_stream tool; the stream_url field is extracted automatically. "
            "For Starlight chat, append ?room=<room_id>&agent=<agent_id> or pass "
            "those as room_id and agent_id."
        ),
        task=StringSchema(
            "REQUIRED when action='start'. Standing instructions for the "
            "subagent that handles each event (who you are, when to reply, "
            "how to use chat_send)."
        ),
        label=StringSchema("Optional short label for display (e.g. 'starlight-contract-abc')."),
        room_id=StringSchema(
            "Optional chat room id. Appended as the 'room' query param when missing from url."
        ),
        agent_id=StringSchema(
            "Optional sender id for this bot. Appended as the 'agent' query param "
            "when missing from url. Events from this id are ignored so the "
            "subagent does not reply to itself."
        ),
        mcp_server=StringSchema(
            "Optional configured MCP server name whose headers (API key) should "
            "be copied onto the SSE request. When omitted, a server whose URL "
            "host matches the stream URL is used."
        ),
        monitor_id=StringSchema(
            "REQUIRED when action='stop'. Monitor id from action='list' or the start result."
        ),
        required=["action"],
        description=(
            "Start, stop, or list background SSE stream monitors. A monitor keeps "
            "one subagent on a live stream (for example a Starlight chat room) so "
            "the main agent can keep working. start requires url and task; stop "
            "requires monitor_id; list only needs action."
        ),
    )
)
class MonitorTool(Tool):
    """Tool to attach a subagent to a live SSE stream."""

    def __init__(self, manager: StreamMonitorManager):
        self._manager = manager

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        return getattr(ctx, "stream_monitors", None) is not None

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(manager=ctx.stream_monitors)

    @property
    def name(self) -> str:
        return "monitor"

    @property
    def description(self) -> str:
        return (
            "Watch a live SSE stream with a background subagent. "
            "Use this for Starlight (and similar) chatrooms: call the MCP "
            "chat_stream tool, then monitor(action=\"start\", url=..., "
            "room_id=..., agent_id=..., task=...). The subagent receives each "
            "event and can reply via MCP chat_send. Actions: start, stop, list."
        )

    async def execute(
        self,
        action: str,
        url: str | None = None,
        task: str | None = None,
        label: str | None = None,
        room_id: str | None = None,
        agent_id: str | None = None,
        mcp_server: str | None = None,
        monitor_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "list":
            return self._manager.format_list()
        if action == "stop":
            if not monitor_id:
                return ToolResult.error("Error: monitor_id is required when action='stop'.")
            return await self._manager.stop(monitor_id)
        if action != "start":
            return ToolResult.error("Error: action must be start, stop, or list.")
        if not url:
            return ToolResult.error("Error: url is required when action='start'.")
        if not task or not str(task).strip():
            return ToolResult.error("Error: task is required when action='start'.")
        request_ctx = current_request_context()
        if request_ctx is None or request_ctx.runtime is None:
            return ToolResult.error("Error: monitor requires an active model runtime")
        session_key = request_ctx.session_key or (
            f"{request_ctx.channel}:{request_ctx.chat_id}"
            if request_ctx.channel and request_ctx.chat_id
            else None
        )
        return await self._manager.start(
            url=url,
            task=task,
            runtime=request_ctx.runtime,
            label=label,
            agent_id=agent_id,
            room_id=room_id,
            mcp_server=mcp_server,
            origin_channel=request_ctx.channel,
            origin_chat_id=request_ctx.chat_id,
            session_key=session_key,
            origin_message_id=request_ctx.message_id,
            workspace_scope=current_workspace_scope(),
        )
