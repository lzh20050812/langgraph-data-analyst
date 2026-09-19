"""Node-level tracing wrapper for LangGraph workflow functions."""

from __future__ import annotations

from functools import wraps
from time import perf_counter
from typing import Callable

from agents.contracts import NodeTrace
from agents.run_context import emit_event, raise_if_cancelled, save_checkpoint
from agents.state import AgentState


def traced_node(name: str, node: Callable[[AgentState], AgentState]):
    @wraps(node)
    def wrapped(state: AgentState) -> AgentState:
        raise_if_cancelled()
        emit_event({"type": "node_started", "node": name})
        started = perf_counter()
        try:
            result = node(state)
        except Exception as exc:
            duration_ms = round((perf_counter() - started) * 1000, 2)
            emit_event({
                "type": "node_failed",
                "node": name,
                "duration_ms": duration_ms,
                "error": type(exc).__name__,
            })
            raise

        duration_ms = round((perf_counter() - started) * 1000, 2)
        failed = bool(result.get("error"))
        trace = NodeTrace(
            node=name,
            status="failed" if failed else "completed",
            duration_ms=duration_ms,
            current_step=result.get("current_step", ""),
            error=result.get("error"),
        ).model_dump(mode="python")
        result.setdefault("execution_trace", []).append(trace)
        emit_event({"type": "node_failed" if failed else "node_completed", **trace})
        save_checkpoint(f"{name}:failed_snapshot" if failed else name, result)
        raise_if_cancelled()
        return result

    return wrapped
