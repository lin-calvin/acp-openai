from __future__ import annotations

from acp.schema import TextContentBlock

from acp_openai_middleware.openai_mapper import (
    to_acp_content_blocks,
    build_non_streaming_response,
    extract_history_messages,
    acp_chunk_to_text,
    chunk_to_delta,
)
from acp_openai_middleware.schemas import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from acp_openai_middleware.session_pool import HistoryMessage


class TestToAcpContentBlocks:
    def test_single_user_message(self):
        messages = [ChatMessage(role="user", content="hello")]
        blocks = to_acp_content_blocks(messages)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContentBlock)
        assert blocks[0].text == "hello"

    def test_system_message_prefixed(self):
        messages = [ChatMessage(role="system", content="be helpful")]
        blocks = to_acp_content_blocks(messages)
        assert len(blocks) == 1
        assert blocks[0].text == "[System]: be helpful"

    def test_mixed_messages(self):
        messages = [
            ChatMessage(role="system", content="you are helpful"),
            ChatMessage(role="user", content="hi"),
        ]
        blocks = to_acp_content_blocks(messages)
        assert len(blocks) == 2
        assert blocks[0].text == "[System]: you are helpful"
        assert blocks[1].text == "hi"

    def test_assistant_messages_included(self):
        messages = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi there"),
            ChatMessage(role="user", content="how are you"),
        ]
        blocks = to_acp_content_blocks(messages)
        assert len(blocks) == 3
        assert blocks[0].text == "hello"
        assert blocks[1].text == "[Assistant]: hi there"
        assert blocks[1].text.startswith("[Assistant]: ")
        assert blocks[2].text == "how are you"

    def test_empty_content(self):
        messages = [ChatMessage(role="user", content=None)]
        blocks = to_acp_content_blocks(messages)
        assert len(blocks) == 1
        assert blocks[0].text == ""

    def test_unknown_role_skipped(self):
        messages = [ChatMessage(role="tool", content="result")]
        blocks = to_acp_content_blocks(messages)
        assert len(blocks) == 0


class TestBuildNonStreamingResponse:
    def test_basic(self):
        response = build_non_streaming_response(
            model="my-model",
            full_text="Hello, how can I help?",
            stop_reason="end_turn",
        )
        assert isinstance(response, ChatCompletionResponse)
        assert response.model == "my-model"
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "Hello, how can I help?"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None

    def test_stop_reason_max_tokens(self):
        response = build_non_streaming_response(
            model="m", full_text="limited", stop_reason="max_tokens"
        )
        assert response.choices[0].finish_reason == "length"

    def test_stop_reason_cancelled(self):
        response = build_non_streaming_response(
            model="m", full_text="", stop_reason="cancelled"
        )
        assert response.choices[0].finish_reason is None

    def test_unknown_stop_reason(self):
        response = build_non_streaming_response(
            model="m", full_text="x", stop_reason="unknown_reason"
        )
        assert response.choices[0].finish_reason == "stop"


class TestExtractHistoryMessages:
    def test_simple(self):
        from acp import text_block
        history = extract_history_messages(
            full_text="I am fine",
            user_blocks=[text_block("how are you")],
        )
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "how are you"
        assert history[1].role == "assistant"
        assert history[1].content == "I am fine"

    def test_no_assistant_text(self):
        from acp import text_block
        history = extract_history_messages(
            full_text="",
            user_blocks=[text_block("hello")],
        )
        assert len(history) == 1
        assert history[0].role == "user"

    def test_assistant_prefix_detected(self):
        from acp import text_block
        history = extract_history_messages(
            full_text="new response",
            user_blocks=[
                text_block("user message"),
                text_block("[Assistant]: prior assistant text"),
                text_block("another user message"),
            ],
        )
        assert len(history) == 4
        assert history[0].role == "user"
        assert history[0].content == "user message"
        assert history[1].role == "assistant"
        assert history[1].content == "prior assistant text"
        assert history[2].role == "user"
        assert history[2].content == "another user message"
        assert history[3].role == "assistant"
        assert history[3].content == "new response"


class TestAcpChunkToText:
    def test_text_chunk(self):
        from acp import update_agent_message_text
        chunk = update_agent_message_text("hello world")
        assert acp_chunk_to_text(chunk) == "hello world"

    def test_non_text_skipped(self):
        from acp.schema import AgentMessageChunk, ImageContentBlock
        chunk = AgentMessageChunk(
            session_update="agent_message_chunk",
            content=ImageContentBlock(type="image", data="base64", mime_type="image/png"),
        )
        assert acp_chunk_to_text(chunk) == ""
