import pytest

from api.query_limiter import QueryLimiter


def test_query_limiter_rejects_excess_work_without_blocking():
    limiter = QueryLimiter(2)

    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.active == 2
    assert limiter.acquire() is False

    limiter.release()
    assert limiter.acquire() is True


def test_query_limiter_rejects_unbalanced_release():
    limiter = QueryLimiter(1)

    with pytest.raises(RuntimeError):
        limiter.release()
