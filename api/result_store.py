"""Thread-safe, bounded storage for request-scoped API results."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from time import monotonic
from typing import Callable


class ResultStore:
    """Keep recent query states isolated by request ID.

    This in-process store is sufficient for the single-process thesis demo. A
    shared store such as Redis is still required when deploying multiple API
    worker processes.
    """

    def __init__(
        self,
        max_entries: int = 100,
        ttl_seconds: float = 3600,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = RLock()

    def _prune_expired(self, now: float) -> None:
        expired = [
            request_id
            for request_id, (created_at, _) in self._items.items()
            if now - created_at >= self.ttl_seconds
        ]
        for request_id in expired:
            self._items.pop(request_id, None)

    def put(self, request_id: str, state: dict) -> None:
        """Store a shallow snapshot and evict expired or oldest entries."""
        now = self._clock()
        with self._lock:
            self._prune_expired(now)
            self._items[request_id] = (now, dict(state))
            self._items.move_to_end(request_id)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def get(self, request_id: str) -> dict | None:
        """Return a snapshot for one request, or ``None`` after expiry."""
        now = self._clock()
        with self._lock:
            self._prune_expired(now)
            item = self._items.get(request_id)
            if item is None:
                return None
            self._items.move_to_end(request_id)
            return dict(item[1])

    def clear(self) -> None:
        """Clear all entries; primarily useful for isolated tests."""
        with self._lock:
            self._items.clear()


result_store = ResultStore()
