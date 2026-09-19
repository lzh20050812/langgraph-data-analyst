"""Request-scoped hooks for workflow events and durable checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator

from agents.runtime_policy import RunBudget


EventSink = Callable[[dict[str, Any]], None]
CheckpointSink = Callable[[str, dict[str, Any]], None]
CancellationCheck = Callable[[], bool]


class WorkflowCancelled(RuntimeError):
    pass

_event_sink: ContextVar[EventSink | None] = ContextVar("event_sink", default=None)
_checkpoint_sink: ContextVar[CheckpointSink | None] = ContextVar(
    "checkpoint_sink", default=None
)
_cancellation_check: ContextVar[CancellationCheck | None] = ContextVar(
    "cancellation_check", default=None
)
_owner_id: ContextVar[str | None] = ContextVar("workflow_owner_id", default=None)
_budget: ContextVar[RunBudget | None] = ContextVar("workflow_budget", default=None)


@contextmanager
def run_hooks(
    *,
    event_sink: EventSink | None = None,
    checkpoint_sink: CheckpointSink | None = None,
    cancellation_check: CancellationCheck | None = None,
    owner_id: str | None = None,
    budget: RunBudget | None = None,
) -> Iterator[None]:
    event_token = _event_sink.set(event_sink)
    checkpoint_token = _checkpoint_sink.set(checkpoint_sink)
    cancellation_token = _cancellation_check.set(cancellation_check)
    owner_token = _owner_id.set(owner_id)
    budget_token = _budget.set(budget)
    try:
        yield
    finally:
        _event_sink.reset(event_token)
        _checkpoint_sink.reset(checkpoint_token)
        _cancellation_check.reset(cancellation_token)
        _owner_id.reset(owner_token)
        _budget.reset(budget_token)


def current_owner_id() -> str | None:
    """Return the authenticated workflow owner for tenant-scoped resources."""
    return _owner_id.get()


def current_budget() -> RunBudget | None:
    return _budget.get()


def emit_event(event: dict[str, Any]) -> None:
    sink = _event_sink.get()
    if sink is not None:
        sink(dict(event))


def save_checkpoint(node: str, state: dict[str, Any]) -> None:
    sink = _checkpoint_sink.get()
    if sink is not None:
        sink(node, dict(state))


def raise_if_cancelled() -> None:
    budget = _budget.get()
    if budget is not None:
        budget.check_duration()
    check = _cancellation_check.get()
    if check is not None and check():
        raise WorkflowCancelled("workflow cancellation requested")
