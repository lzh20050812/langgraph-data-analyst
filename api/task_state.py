"""Canonical task states and guarded lifecycle transitions."""

from __future__ import annotations


TASK_STATUSES = frozenset({
    "queued",
    "running",
    "waiting_clarification",
    "cancelling",
    "completed",
    "failed",
    "cancelled",
    # Compatibility terminal for work abandoned by a dead embedded process or
    # an exhausted external-worker lease. A user may restart it explicitly.
    "interrupted",
})

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})

LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "waiting_clarification", "cancelled", "interrupted"}),
    "running": frozenset({
        "queued",  # expired lease hand-off
        "waiting_clarification",
        "cancelling",
        "completed",
        "failed",
        "cancelled",
        "interrupted",
    }),
    "waiting_clarification": frozenset({"queued", "cancelled", "interrupted"}),
    "cancelling": frozenset({"cancelled", "interrupted"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "interrupted": frozenset(),
}


class InvalidTaskTransition(RuntimeError):
    """Raised when a caller attempts to bypass the task state machine."""


def ensure_transition(current: str, target: str) -> None:
    if current not in TASK_STATUSES:
        raise InvalidTaskTransition(f"unknown current task status: {current}")
    if target not in TASK_STATUSES:
        raise InvalidTaskTransition(f"unknown target task status: {target}")
    if target == current:
        return
    if target not in LEGAL_TRANSITIONS[current]:
        raise InvalidTaskTransition(f"illegal task transition: {current} -> {target}")
