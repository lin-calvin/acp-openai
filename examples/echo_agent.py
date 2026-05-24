from __future__ import annotations

import asyncio
import logging
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message,
)
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    Implementation,
    PromptCapabilities,
)

logger = logging.getLogger(__name__)


class EchoAgent(Agent):
    def __init__(self) -> None:
        self._next_session_id = 0
        self._sessions: set[str] = set()
        self._conn = None

    def on_connect(self, conn) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(image=True, embedded_context=True),
            ),
            agent_info=Implementation(
                name="echo-agent",
                title="Echo Agent",
                version="0.1.0",
            ),
        )

    async def new_session(
        self, cwd: str, mcp_servers: list = None, **kwargs: Any
    ) -> NewSessionResponse:
        session_id = f"sess_{self._next_session_id}"
        self._next_session_id += 1
        self._sessions.add(session_id)
        logger.info("new session: %s", session_id)
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self, prompt: list, session_id: str, message_id: str | None = None, **kwargs: Any
    ) -> PromptResponse:
        logger.info("prompt for %s with %d blocks", session_id, len(prompt))
        for block in prompt:
            if block.type == "text":
                await self._conn.session_update(
                    session_id, update_agent_message(text_block(f"Echo: {block.text}"))
                )
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        logger.info("cancel: %s", session_id)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    await run_agent(EchoAgent())


if __name__ == "__main__":
    asyncio.run(main())
