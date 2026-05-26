from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import hashlib
import json
import logging
import os
import sys
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Client,
    text_block,
)
from acp.client.connection import ClientSideConnection
from acp.exceptions import RequestError
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ImageContentBlock,
    Implementation,
    PermissionOption,
    PromptResponse,
    ResourceContentBlock,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    CreateTerminalResponse,
    KillTerminalResponse,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
    SessionInfoUpdate,
    AgentPlanUpdate,
    AvailableCommandsUpdate,
    CurrentModeUpdate,
    ConfigOptionUpdate,
    UsageUpdate,
    UserMessageChunk,
)

from .openai_mapper import ACP_CONTENT_BLOCK, acp_chunk_to_text
from .session_pool import SessionEntry, SessionPool, HistoryMessage

logger = logging.getLogger(__name__)


def _tool_call_content_to_text(contents: list) -> str:
    parts: list[str] = []
    for c in contents or []:
        if hasattr(c, "type") and c.type == "content":
            if isinstance(c.content, TextContentBlock):
                parts.append(c.content.text)
        elif hasattr(c, "type") and c.type == "diff":
            parts.append(f"{c.path}: {c.old_text or ''} → {c.new_text}")
        elif hasattr(c, "type") and c.type == "terminal":
            parts.append(f"terminal_id={c.terminal_id}")
    return " | ".join(parts)


def _format_tool_call(tc) -> str:
    title = getattr(tc, "title", None) or "unknown"
    kind = getattr(tc, "kind", None) or ""
    status = getattr(tc, "status", None) or ""

    lines = [f"Tool call: {title}"]
    if kind:
        lines.append(f"  kind: {kind}")
    if status:
        lines.append(f"  status: {status}")

    raw_input = getattr(tc, "raw_input", None)
    if raw_input is not None:
        if isinstance(raw_input, (dict, list)):
            lines.append("  input:")
            lines.append(_format_yaml_value(raw_input, indent=1))
        else:
            lines.append(f"  input: {raw_input}")

    raw_output = getattr(tc, "raw_output", None)
    if raw_output is not None:
        if isinstance(raw_output, (dict, list)):
            lines.append("  output:")
            lines.append(_format_yaml_value(raw_output, indent=1))
        else:
            lines.append(f"  output: {raw_output}")

    content_text = _tool_call_content_to_text(getattr(tc, "content", []) or [])
    if content_text:
        lines.append(f"  content: {content_text}")

    return "\n".join(lines)


def _format_yaml_value(value: Any, indent: int = 0) -> str:
    child_prefix = "  " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        parts: list[str] = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                parts.append(f"{child_prefix}{k}:")
                parts.append(_format_yaml_value(v, indent + 1))
            else:
                parts.append(f"{child_prefix}{k}: {v}")
        return "\n".join(parts)
    elif isinstance(value, list):
        if not value:
            return "[]"
        parts: list[str] = []
        for item in value:
            if isinstance(item, (dict, list)):
                parts.append(f"{child_prefix}-")
                parts.append(_format_yaml_value(item, indent + 1))
            else:
                parts.append(f"{child_prefix}- {item}")
        return "\n".join(parts)
    return str(value)


def _build_think_block(think_parts: list[str]) -> str:
    if not think_parts:
        return ""
    return "<think>\n" + "\n".join(think_parts) + "\n</think>\n\n"


class _MiddlewareClient(Client):
    def __init__(self, manager: NamespaceState) -> None:
        self._manager = manager
        self._pending_chunks: list[AgentMessageChunk] = []

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallStart | ToolCallProgress,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        raise RequestError.method_not_found("session/request_permission")

    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate
        | ConfigOptionUpdate
        | SessionInfoUpdate
        | UsageUpdate,
        **kwargs: Any,
    ) -> None:
        if isinstance(update, AgentMessageChunk):
            self._manager._pending_chunks.append(update)
        elif isinstance(update, UsageUpdate):
            self._manager._last_usage = update
        elif isinstance(update, (ToolCallStart, ToolCallProgress)):
            think_text = _format_tool_call(update)
            if think_text:
                self._manager._think_parts.append(think_text)
        elif isinstance(update, AgentThoughtChunk):
            if isinstance(update.content, TextContentBlock):
                self._manager._think_parts.append(update.content.text)

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        raise RequestError.method_not_found("fs/read_text_file")

    async def write_text_file(
        self,
        content: str,
        path: str,
        session_id: str,
        **kwargs: Any,
    ) -> WriteTextFileResponse | None:
        raise RequestError.method_not_found("fs/write_text_file")

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalResponse | None:
        raise RequestError.method_not_found("terminal/kill")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        raise RequestError.method_not_found(method)


class NamespaceState:
    def __init__(
        self,
        conn: ClientSideConnection,
        process: aio_subprocess.Process,
        agent_name: str,
        cwd: str,
        ttl: float,
        max_sessions: int,
    ) -> None:
        self.conn = conn
        self.process = process
        self.agent_name = agent_name
        self.cwd = cwd
        self.pool = SessionPool(ttl_seconds=ttl, max_sessions=max_sessions)
        self.lock = asyncio.Lock()
        self._pending_chunks: list[AgentMessageChunk] = []
        self._last_usage: UsageUpdate | None = None
        self._think_parts: list[str] = []

    def clear_pending(self) -> None:
        self._pending_chunks = []
        self._last_usage = None
        self._think_parts = []


class AgentSessionManager:
    def __init__(self, agent_command: str, agent_args: list[str], cwd: str, env: dict[str, str] | None = None,
                 session_ttl: float = 3600, max_sessions: int = 50) -> None:
        self._agent_command = agent_command
        self._agent_args = agent_args
        self._cwd = cwd
        self._env = env or {}
        self._session_ttl = session_ttl
        self._max_sessions = max_sessions
        self._namespaces: dict[str, NamespaceState] = {}
        self._global_lock = asyncio.Lock()

        # Resolve the real command
        parsed = self._resolve_command()
        self._resolved_command = parsed[0]
        self._resolved_args = parsed[1:]

    def _resolve_command(self) -> list[str]:
        cmd = self._agent_command
        if cmd in ("python", "python3"):
            return [sys.executable, *self._agent_args]
        if cmd.endswith(".py"):
            return [sys.executable, cmd, *self._agent_args]
        parts = cmd.split()
        if len(parts) > 1:
            return parts + self._agent_args
        return [cmd, *self._agent_args]

    def _hash_key(self, api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()[:32]

    async def get_or_create_namespace(self, api_key: str, api_key_raw: str = "") -> NamespaceState:
        ns_key = self._hash_key(api_key)
        async with self._global_lock:
            ns = self._namespaces.get(ns_key)
            if ns is not None:
                if ns.process.returncode is not None:
                    logger.warning("Agent process for namespace %s exited, recreating", ns_key)
                    self._namespaces.pop(ns_key)
                    ns = None

            if ns is None:
                ns = await self._spawn_namespace(ns_key)
                self._namespaces[ns_key] = ns

            return ns

    async def _spawn_namespace(self, ns_key: str) -> NamespaceState:
        cmd_parts = [self._resolved_command, *self._resolved_args]
        logger.info("Spawning agent for namespace %s: %s", ns_key, cmd_parts)

        full_env = {**os.environ, **self._env}

        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdin=aio_subprocess.PIPE,
            stdout=aio_subprocess.PIPE,
            stderr=aio_subprocess.PIPE,
            env=full_env,
            cwd=self._cwd,
        )

        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Agent process does not expose stdio pipes")

        # Start stderr reader so agent doesn't block on stderr buffer
        async def _read_stderr() -> None:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                logger.debug("[agent stderr] %s", line.decode(errors="replace").rstrip())

        _ = asyncio.create_task(_read_stderr())

        client_impl = _MiddlewareClient(manager=None)  # placeholder, set after
        conn = ClientSideConnection(client_impl, proc.stdin, proc.stdout)

        await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
            client_info=Implementation(
                name="acp-openai-middleware", title="ACP OpenAI Middleware", version="0.1.0"
            ),
        )

        state = NamespaceState(
            conn=conn,
            process=proc,
            agent_name=f"acp-agent-{ns_key[:8]}",
            cwd=self._cwd,
            ttl=self._session_ttl,
            max_sessions=self._max_sessions,
        )

        # Re-point the client impl to the now-created state
        client_impl._manager = state

        return state

    async def _new_agent_session(self, ns: NamespaceState, cwd: str) -> str:
        result = await ns.conn.new_session(cwd=cwd, mcp_servers=[])
        return result.session_id

    async def find_or_create_session(
        self, ns: NamespaceState, messages: list[Any], cwd: str
    ) -> tuple[SessionEntry, int]:
        return await ns.pool.find_or_create(
            messages,
            lambda cwd: self._new_agent_session(ns, cwd),
            cwd,
        )

    async def send_prompt(
        self,
        ns: NamespaceState,
        session_id: str,
        content_blocks: list[ACP_CONTENT_BLOCK],
    ) -> tuple[str, PromptResponse]:
        ns.clear_pending()
        try:
            response = await ns.conn.prompt(
                session_id=session_id,
                prompt=content_blocks,
                message_id=None,
            )
        except Exception:
            # HACK: opencode returns negative outputTokens (e.g. -13), which violates
            # the ACP schema validation (ge=0). We catch the resulting Pydantic
            # ValidationError here and return whatever content chunks arrived via
            # session/update notifications before the prompt response blew up.
            logger.exception(
                "Prompt response validation failed for session %s "
                "(likely negative token counts from agent), returning partial content",
                session_id,
            )
            full_text = "".join(acp_chunk_to_text(c) for c in ns._pending_chunks)
            response = PromptResponse(stop_reason="end_turn")
            think_block = _build_think_block(ns._think_parts)
            return think_block + full_text, response
        full_text = "".join(acp_chunk_to_text(c) for c in ns._pending_chunks)
        think_block = _build_think_block(ns._think_parts)
        return think_block + full_text, response

    async def evict_stale_from_all(self) -> int:
        total = 0
        async with self._global_lock:
            for ns in list(self._namespaces.values()):
                total += await ns.pool.evict_stale()
        return total

    async def close_all(self) -> None:
        async with self._global_lock:
            for ns_key, ns in list(self._namespaces.items()):
                try:
                    if ns.process.returncode is None:
                        ns.process.terminate()
                        try:
                            await asyncio.wait_for(ns.process.wait(), timeout=5)
                        except asyncio.TimeoutError:
                            ns.process.kill()
                            await ns.process.wait()
                except Exception:
                    logger.exception("Error closing agent for namespace %s", ns_key)
                await ns.conn.close()
            self._namespaces.clear()
