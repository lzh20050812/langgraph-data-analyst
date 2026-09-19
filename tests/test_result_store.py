from api.result_store import ResultStore


def test_results_are_isolated_by_request_id():
    store = ResultStore()
    store.put("request-a", {"report": "A", "charts": [{"id": "a"}]})
    store.put("request-b", {"report": "B", "charts": [{"id": "b"}]})

    assert store.get("request-a")["report"] == "A"
    assert store.get("request-b")["report"] == "B"


def test_store_expires_old_results():
    now = [100.0]
    store = ResultStore(ttl_seconds=10, clock=lambda: now[0])
    store.put("request-a", {"report": "A"})

    now[0] = 109.9
    assert store.get("request-a") is not None

    now[0] = 110.0
    assert store.get("request-a") is None


def test_store_evicts_the_oldest_result_at_capacity():
    store = ResultStore(max_entries=2)
    store.put("request-a", {"report": "A"})
    store.put("request-b", {"report": "B"})
    store.put("request-c", {"report": "C"})

    assert store.get("request-a") is None
    assert store.get("request-b")["report"] == "B"
    assert store.get("request-c")["report"] == "C"
