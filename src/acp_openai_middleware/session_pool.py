from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import ChatMessage


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    role: str
    content: str

    def matches(self, msg: ChatMessage) -> bool:
        if self.role != msg.role:
            return False
        return _normalize(self.content) == _normalize(msg.content)

    @classmethod
    def from_chat_message(cls, msg: ChatMessage) -> HistoryMessage:
        return cls(role=msg.role, content=_normalize(msg.content))


def _match_prefix_length(
    history: list[HistoryMessage], messages: list[ChatMessage]
) -> int:
    n = 0
    limit = min(len(history), len(messages))
    while n < limit and history[n].matches(messages[n]):
        n += 1
    return n


@dataclass
class SessionEntry:
    session_id: str
    message_history: list[HistoryMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used_at = time.time()


@dataclass
class PoolLookupResult:
    session: SessionEntry
    prefix_len: int


class SessionPool:
    def __init__(self, ttl_seconds: float = 3600, max_sessions: int = 50):
        self._sessions: dict[str, SessionEntry] = {}
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._lock = asyncio.Lock()

    async def find_or_create(
        self, messages: list[ChatMessage], session_factory, cwd: str
    ) -> PoolLookupResult:
        async with self._lock:
            best = self._find_best(messages)
            if best is not None:
                best.session.touch()
                return best

            if len(self._sessions) >= self._max:
                await self._evict_one()

            session_id = await session_factory(cwd)
            entry = SessionEntry(session_id=session_id)
            self._sessions[session_id] = entry
            return PoolLookupResult(session=entry, prefix_len=0)

    def _find_best(self, messages: list[ChatMessage]) -> PoolLookupResult | None:
        best_session: SessionEntry | None = None
        best_len = 0
        best_time = 0.0

        for entry in self._sessions.values():
            match_len = _match_prefix_length(entry.message_history, messages)
            if match_len == 0:
                continue
            if match_len > best_len:
                best_len = match_len
                best_session = entry
                best_time = entry.last_used_at
            elif match_len == best_len and entry.last_used_at > best_time:
                best_session = entry
                best_time = entry.last_used_at

        if best_session is None:
            return None
        return PoolLookupResult(session=best_session, prefix_len=best_len)

    async def record_response(
        self, session_id: str, new_messages: list[HistoryMessage]
    ) -> None:
        async with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return
            entry.message_history.extend(new_messages)
            entry.touch()

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def evict_stale(self) -> int:
        async with self._lock:
            now = time.time()
            stale = [
                sid
                for sid, entry in self._sessions.items()
                if now - entry.last_used_at > self._ttl
            ]
            for sid in stale:
                self._sessions.pop(sid, None)
            return len(stale)

    async def _evict_one(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.items(), key=lambda kv: kv[1].last_used_at)
        self._sessions.pop(oldest[0], None)

    def __len__(self) -> int:
        return len(self._sessions)
