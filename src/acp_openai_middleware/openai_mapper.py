from __future__ import annotations

import time
from uuid import uuid4

from acp import text_block
from acp.schema import (
    AgentMessageChunk,
    TextContentBlock,
)

from .schemas import (
    ChatCompletionChoice,
    ChatCompletionDelta,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from .session_pool import HistoryMessage

ACP_CONTENT_BLOCK = TextContentBlock


_ROLE_PREFIX = "[Assistant]: "


def to_acp_content_blocks(
    messages: list[ChatMessage],
) -> list[ACP_CONTENT_BLOCK]:
    blocks: list[ACP_CONTENT_BLOCK] = []
    for msg in messages:
        content = msg.content or ""
        if msg.role == "system":
            blocks.append(text_block(f"[System]: {content}"))
        elif msg.role == "user":
            blocks.append(text_block(content))
        elif msg.role == "assistant":
            blocks.append(text_block(f"{_ROLE_PREFIX}{content}"))
    return blocks


def acp_chunk_to_text(chunk: AgentMessageChunk) -> str:
    if isinstance(chunk.content, TextContentBlock):
        return chunk.content.text
    return ""


def chunk_to_delta(chunk: AgentMessageChunk) -> ChatCompletionDelta | None:
    text = acp_chunk_to_text(chunk)
    if not text:
        return None
    return ChatCompletionDelta(content=text)


def build_non_streaming_response(
    model: str,
    full_text: str,
    stop_reason: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> ChatCompletionResponse:
    finish_reason = _map_stop_reason(stop_reason)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex[:12]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=full_text),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def extract_history_messages(
    full_text: str,
    user_blocks: list[ACP_CONTENT_BLOCK],
) -> list[HistoryMessage]:
    result: list[HistoryMessage] = []
    for block in user_blocks:
        if isinstance(block, TextContentBlock):
            if block.text.startswith(_ROLE_PREFIX):
                content = block.text[len(_ROLE_PREFIX):]
                result.append(HistoryMessage(role="assistant", content=content))
            else:
                result.append(HistoryMessage(role="user", content=block.text))
    if full_text:
        result.append(HistoryMessage(role="assistant", content=full_text))
    return result


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
