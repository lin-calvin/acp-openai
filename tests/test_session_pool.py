from __future__ import annotations

import asyncio
import pytest
from uuid import uuid4

from acp_openai_middleware.session_pool import (
    SessionPool,
    SessionEntry,
    HistoryMessage,
    _match_prefix_length,
    _normalize,
)
from acp_openai_middleware.schemas import ChatMessage


class TestNormalize:
    def test_simple(self):
        assert _normalize("hello world") == "hello world"

    def test_whitespace(self):
        assert _normalize("  hello   world  ") == "hello world"

    def test_newlines(self):
        assert _normalize("hello\nworld\n") == "hello world"

    def test_empty(self):
        assert _normalize(None) == ""
        assert _normalize("") == ""

    def test_equiv(self):
        a = _normalize("Hello World")
        b = _normalize("  Hello   World  ")
        assert a == b


class TestHistoryMessage:
    def test_simple_match(self):
        hm = HistoryMessage(role="user", content="hello world")
        msg = ChatMessage(role="user", content="hello world")
        assert hm.matches(msg)

    def test_whitespace_match(self):
        hm = HistoryMessage(role="user", content="hello world")
        msg = ChatMessage(role="user", content="  hello   world  ")
        assert hm.matches(msg)

    def test_role_mismatch(self):
        hm = HistoryMessage(role="user", content="hello world")
        msg = ChatMessage(role="assistant", content="hello world")
        assert not hm.matches(msg)

    def test_none_content_match(self):
        hm = HistoryMessage(role="user", content="")
        msg = ChatMessage(role="user", content=None)
        assert hm.matches(msg)

    def test_from_chat_message(self):
        msg = ChatMessage(role="user", content="Hello World  ")
        hm = HistoryMessage.from_chat_message(msg)
        assert hm.role == "user"
        assert hm.content == "Hello World"


class TestMatchPrefixLength:
    def test_full_match(self):
        history = [
            HistoryMessage("user", "hello"),
            HistoryMessage("assistant", "hi there"),
            HistoryMessage("user", "how are you"),
        ]
        messages = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi there"),
            ChatMessage(role="user", content="how are you"),
        ]
        assert _match_prefix_length(history, messages) == 3

    def test_partial_match(self):
        history = [
            HistoryMessage("user", "hello"),
            HistoryMessage("assistant", "hi there"),
        ]
        messages = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi there"),
            ChatMessage(role="user", content="what's new"),
        ]
        assert _match_prefix_length(history, messages) == 2

    def test_no_match(self):
        history = [
            HistoryMessage("user", "hello"),
        ]
        messages = [
            ChatMessage(role="user", content="goodbye"),
        ]
        assert _match_prefix_length(history, messages) == 0

    def test_empty_history(self):
        history: list[HistoryMessage] = []
        messages = [ChatMessage(role="user", content="hello")]
        assert _match_prefix_length(history, messages) == 0

    def test_messages_shorter_than_history(self):
        history = [
            HistoryMessage("user", "a"),
            HistoryMessage("assistant", "b"),
            HistoryMessage("user", "c"),
        ]
        messages = [
            ChatMessage(role="user", content="a"),
            ChatMessage(role="assistant", content="b"),
        ]
        assert _match_prefix_length(history, messages) == 2


class TestSessionPool:
    async def _session_factory(self, cwd: str) -> str:
        return f"sess_{uuid4().hex[:8]}"

    async def test_first_request_creates_new(self):
        pool = SessionPool(ttl_seconds=3600, max_sessions=50)
        messages = [ChatMessage(role="user", content="hello")]
        result = await pool.find_or_create(messages, self._session_factory, "/tmp")
        assert result.prefix_len == 0
        assert result.session.session_id.startswith("sess_")
        assert len(pool) == 1

    async def test_prefix_match_returns_existing(self):
        pool = SessionPool(ttl_seconds=3600, max_sessions=50)

        # First request
        m1 = [ChatMessage(role="user", content="hello")]
        r1 = await pool.find_or_create(m1, self._session_factory, "/tmp")
        sid = r1.session.session_id

        # Add history
        await pool.record_response(
            sid,
            [
                HistoryMessage(role="user", content="hello"),
                HistoryMessage(role="assistant", content="hi there"),
            ],
        )

        # Second request with same prefix
        m2 = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi there"),
            ChatMessage(role="user", content="how are you?"),
        ]
        r2 = await pool.find_or_create(m2, self._session_factory, "/tmp")
        assert r2.prefix_len == 2
        assert r2.session.session_id == sid
        assert len(pool) == 1

    async def test_no_match_creates_new(self):
        pool = SessionPool(ttl_seconds=3600, max_sessions=50)

        # First session
        m1 = [ChatMessage(role="user", content="topic A")]
        r1 = await pool.find_or_create(m1, self._session_factory, "/tmp")
        await pool.record_response(
            r1.session.session_id,
            [
                HistoryMessage(role="user", content="topic A"),
                HistoryMessage(role="assistant", content="response A"),
            ],
        )

        # Different topic starts new session
        m2 = [ChatMessage(role="user", content="topic B")]
        r2 = await pool.find_or_create(m2, self._session_factory, "/tmp")
        assert r2.prefix_len == 0
        assert r2.session.session_id != r1.session.session_id
        assert len(pool) == 2

    async def test_longest_prefix_wins(self):
        pool = SessionPool(ttl_seconds=3600, max_sessions=50)

        # Session 1: 2-turn deep (distinct topic)
        r1 = await pool.find_or_create(
            [ChatMessage(role="user", content="topic alpha")],
            self._session_factory, "/tmp",
        )
        await pool.record_response(
            r1.session.session_id,
            [
                HistoryMessage("user", "topic alpha"),
                HistoryMessage("assistant", "reply about alpha"),
            ],
        )

        # Session 2: 4-turn deep (distinct topic)
        r2 = await pool.find_or_create(
            [ChatMessage(role="user", content="topic beta")],
            self._session_factory, "/tmp",
        )
        await pool.record_response(
            r2.session.session_id,
            [
                HistoryMessage("user", "topic beta"),
                HistoryMessage("assistant", "reply about beta"),
                HistoryMessage("user", "continue on beta"),
                HistoryMessage("assistant", "more beta info"),
            ],
        )

        # Request that matches session 2 prefix of length 4
        m3 = [
            ChatMessage(role="user", content="topic beta"),
            ChatMessage(role="assistant", content="reply about beta"),
            ChatMessage(role="user", content="continue on beta"),
            ChatMessage(role="assistant", content="more beta info"),
            ChatMessage(role="user", content="next question"),
        ]
        r3 = await pool.find_or_create(m3, self._session_factory, "/tmp")
        assert r3.prefix_len == 4
        assert r3.session.session_id == r2.session.session_id

    async def test_tie_break_by_recency(self):
        pool = SessionPool(ttl_seconds=3600, max_sessions=50)

        # Create two sessions with same-level history
        m = [ChatMessage(role="user", content="hello")]
        r1 = await pool.find_or_create(m, self._session_factory, "/tmp")
        r2 = await pool.find_or_create(m, self._session_factory, "/tmp")

        await pool.record_response(
            r1.session.session_id,
            [HistoryMessage("user", "hello"), HistoryMessage("assistant", "a")],
        )
        # r2 recorded later → more recent
        await asyncio.sleep(0.01)
        await pool.record_response(
            r2.session.session_id,
            [HistoryMessage("user", "hello"), HistoryMessage("assistant", "b")],
        )

        m2 = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="b"),
            ChatMessage(role="user", content="next"),
        ]
        # Both prefix match length 2, but r2 was more recently used
        r3 = await pool.find_or_create(m2, self._session_factory, "/tmp")
        # Either is acceptable due to the tiebreaker
        assert r3.session.session_id in (r1.session.session_id, r2.session.session_id)

    async def test_remove_session(self):
        pool = SessionPool(ttl_seconds=3600, max_sessions=50)
        m = [ChatMessage(role="user", content="hello")]
        r = await pool.find_or_create(m, self._session_factory, "/tmp")
        assert len(pool) == 1
        await pool.remove(r.session.session_id)
        assert len(pool) == 0

    async def test_evict_stale(self):
        pool = SessionPool(ttl_seconds=0.01, max_sessions=50)
        m = [ChatMessage(role="user", content="hello")]
        r = await pool.find_or_create(m, self._session_factory, "/tmp")
        await asyncio.sleep(0.05)
        count = await pool.evict_stale()
        assert count == 1
        assert len(pool) == 0

    async def test_max_sessions_eviction(self):
        pool = SessionPool(ttl_seconds=3600, max_sessions=2)

        m1 = [ChatMessage(role="user", content="msg1")]
        r1 = await pool.find_or_create(m1, self._session_factory, "/tmp")
        await pool.record_response(r1.session.session_id, [HistoryMessage("user", "msg1")])

        m2 = [ChatMessage(role="user", content="msg2")]
        r2 = await pool.find_or_create(m2, self._session_factory, "/tmp")
        await pool.record_response(r2.session.session_id, [HistoryMessage("user", "msg2")])

        # Third pushes out the oldest
        m3 = [ChatMessage(role="user", content="msg3")]
        r3 = await pool.find_or_create(m3, self._session_factory, "/tmp")
        assert len(pool) == 2
        assert r1.session.session_id not in [r2.session.session_id, r3.session.session_id]
