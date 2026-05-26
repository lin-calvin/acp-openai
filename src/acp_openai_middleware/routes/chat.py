from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ..agent_manager import AgentSessionManager, NamespaceState, _build_think_block
from ..openai_mapper import (
    ACP_CONTENT_BLOCK,
    build_non_streaming_response,
    extract_history_messages,
    to_acp_content_blocks,
)
from ..schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from ..session_pool import HistoryMessage, SessionEntry

logger = logging.getLogger(__name__)

router = APIRouter()


def _check_auth(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    key = auth[7:]
    allowed = request.app.state.api_keys
    if allowed and key not in allowed:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


_STOP_MAP: dict[str, str | None] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "max_turn_requests": "length",
    "refusal": "stop",
    "cancelled": None,
}


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    manager: AgentSessionManager = request.app.state.agent_manager
    model = body.model or request.app.state.agent_name

    api_key = _check_auth(request)
    ns = await manager.get_or_create_namespace(api_key)

    messages = body.messages
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    async with ns.lock:
        result = await manager.find_or_create_session(ns, messages, ns.cwd)
        entry = result.session
        prefix_len = result.prefix_len
        new_messages = messages[prefix_len:]
        content_blocks = to_acp_content_blocks(new_messages)
        completion_id = f"chatcmpl-{uuid4().hex[:12]}"

        if body.stream:
            return EventSourceResponse(
                _stream_events(manager, ns, entry, content_blocks, completion_id, model)
            )

        full_text, response = await manager.send_prompt(
            ns, entry.session_id, content_blocks
        )

    stop_reason_raw = response.stop_reason

    result = build_non_streaming_response(
        model=model,
        full_text=full_text,
        stop_reason=stop_reason_raw,
    )
    result.id = completion_id

    history_msgs = extract_history_messages(full_text, content_blocks)
    await ns.pool.record_response(entry.session_id, history_msgs)

    return result


async def _stream_events(
    manager: AgentSessionManager,
    ns: NamespaceState,
    entry: SessionEntry,
    content_blocks: list[ACP_CONTENT_BLOCK],
    completion_id: str,
    model: str,
):
    ns.clear_pending()

    prompt_task = asyncio.create_task(
        ns.conn.prompt(
            session_id=entry.session_id,
            prompt=content_blocks,
            message_id=None,
        )
    )

    sent_role = False
    think_sent = False
    finish_reason: str | None = None

    from ..openai_mapper import acp_chunk_to_text, chunk_to_delta

    full_text_parts: list[str] = []

    while not prompt_task.done() or ns._pending_chunks:
        if not think_sent and ns._think_parts:
            think_block = _build_think_block(ns._think_parts)
            delta = ChatCompletionDelta(content=think_block, role="assistant")
            sent_role = True
            think_sent = True
            full_text_parts.append(think_block)
            payload = ChatCompletionChunk(
                id=completion_id,
                created=int(time.time()),
                model=model,
                choices=[ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=None)],
            ).model_dump(mode="json", exclude_none=True)
            yield {"data": json.dumps(payload)}

        while ns._pending_chunks:
            chunk = ns._pending_chunks.pop(0)
            text = acp_chunk_to_text(chunk)
            if text:
                full_text_parts.append(text)
            delta = chunk_to_delta(chunk)
            if delta is None:
                continue
            if not sent_role:
                delta.role = "assistant"
                sent_role = True
            payload = ChatCompletionChunk(
                id=completion_id,
                created=int(time.time()),
                model=model,
                choices=[ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=None)],
            ).model_dump(mode="json", exclude_none=True)
            yield {"data": json.dumps(payload)}

        if prompt_task.done():
            break
        await asyncio.sleep(0.01)

    if prompt_task.done():
        try:
            response = prompt_task.result()
        except Exception:
            # HACK: opencode returns negative outputTokens which violates the
            # ACP schema (ge=0). Content chunks already arrived via
            # session/update notifications, so we yield a partial response.
            logger.exception(
                "Prompt response validation failed for session %s "
                "(likely negative token counts from agent), yielding partial stream",
                entry.session_id,
            )
            finish_reason = None
        else:
            sr = response.stop_reason
            finish_reason = _STOP_MAP.get(sr, "stop")
    else:
        finish_reason = None

    final_payload = ChatCompletionChunk(
        id=completion_id,
        created=int(time.time()),
        model=model,
        choices=[ChatCompletionChunkChoice(
            index=0,
            delta=ChatCompletionDelta(),
            finish_reason=finish_reason,
        )],
    ).model_dump(mode="json", exclude_none=True)
    yield {"data": json.dumps(final_payload)}
    yield {"data": "[DONE]"}

    think_block = _build_think_block(ns._think_parts) if not think_sent else ""
    full_text = think_block + "".join(full_text_parts)
    history_msgs = extract_history_messages(full_text, content_blocks)
    await ns.pool.record_response(entry.session_id, history_msgs)
