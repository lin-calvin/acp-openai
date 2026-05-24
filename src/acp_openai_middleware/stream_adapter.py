from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

from acp.schema import (
    AgentMessageChunk,
    PromptResponse,
    TextContentBlock,
    UsageUpdate,
)

from .openai_mapper import ACP_CONTENT_BLOCK, chunk_to_delta
from .schemas import ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta


class StreamCollector:
    def __init__(self, model: str, completion_id: str) -> None:
        self.model = model
        self.completion_id = completion_id
        self.chunks: list[AgentMessageChunk] = []
        self.usage: UsageUpdate | None = None
        self.stop_reason: str | None = None
        self._done = asyncio.Event()

    def on_chunk(self, chunk: AgentMessageChunk) -> None:
        self.chunks.append(chunk)

    def on_usage(self, usage: UsageUpdate) -> None:
        self.usage = usage

    def on_done(self, stop_reason: str | None) -> None:
        self.stop_reason = stop_reason
        self._done.set()

    def full_text(self) -> str:
        return "".join(
            chunk.content.text
            for chunk in self.chunks
            if isinstance(chunk.content, TextContentBlock)
        )


async def acp_to_sse_stream(
    send_prompt_fn,
    collector: StreamCollector,
    session_id: str,
    content_blocks: list[ACP_CONTENT_BLOCK],
    model: str,
) -> str:
    completion_id = collector.completion_id

    # Wrap the collector's callback into the manager's pending pipeline
    # The agent_manager's _MiddlewareClient.session_update will call on_chunk

    # Start prompt (non-blocking collect via the client callback)
    prompt_task = asyncio.create_task(
        send_prompt_fn(session_id, content_blocks)
    )

    sent_role = False
    while not prompt_task.done() or collector.chunks:
        # Yield any buffered chunks
        while collector.chunks:
            chunk = collector.chunks.pop(0)
            delta = chunk_to_delta(chunk)
            if delta is None:
                continue

            if not sent_role:
                delta.role = "assistant"
                sent_role = True

            chunk_json = ChatCompletionChunk(
                id=completion_id,
                created=int(time.time()),
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=delta,
                        finish_reason=None,
                    )
                ],
            ).model_dump(mode="json", exclude_none=True)
            yield f"data: {json.dumps(chunk_json)}\n\n"

        if prompt_task.done():
            break

        await asyncio.sleep(0.01)

    # Handle prompt response
    if prompt_task.done():
        full_text, response = prompt_task.result()
        finish_reason = _map_stop_reason(response.stop_reason)
    else:
        full_text = collector.full_text()
        finish_reason = _map_stop_reason(collector.stop_reason)

    final_delta = ChatCompletionDelta()
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=final_delta,
                finish_reason=finish_reason,
            )
        ],
    ).model_dump(mode="json", exclude_none=True)
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _map_stop_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "max_turn_requests": "length",
        "refusal": "stop",
        "cancelled": None,
    }
    return mapping.get(reason, "stop")


async def acp_streaming_wrapper(
    send_prompt_fn,
    collector: StreamCollector,
    session_id: str,
    content_blocks: list[ACP_CONTENT_BLOCK],
    model: str,
) -> str:
    result_generator = acp_to_sse_stream(
        send_prompt_fn, collector, session_id, content_blocks, model
    )
    full_text: str = collector.full_text()
    async for _event in result_generator:
        pass
    return full_text
