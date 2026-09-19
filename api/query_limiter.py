"""Non-blocking concurrency guard for expensive analysis requests."""

from __future__ import annotations

from threading import BoundedSemaphore, Lock

from config.settings import get_settings


class QueryLimiter:
    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.max_concurrent = max_concurrent
        self._semaphore = BoundedSemaphore(max_concurrent)
        self._lock = Lock()
        self._active = 0

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def acquire(self) -> bool:
        acquired = self._semaphore.acquire(blocking=False)
        if acquired:
            with self._lock:
                self._active += 1
        return acquired

    def release(self) -> None:
        with self._lock:
            if self._active < 1:
                raise RuntimeError("query limiter released without an active query")
            self._active -= 1
        self._semaphore.release()


query_limiter = QueryLimiter(get_settings().MAX_CONCURRENT_QUERIES)
