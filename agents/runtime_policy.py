"""Workflow budgets and stable, non-sensitive failure classification."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any


class WorkflowBudgetExceeded(RuntimeError):
    """A configured task-level resource budget was exhausted."""


@dataclass
class RunBudget:
    max_calls: int
    max_context_chars: int
    max_total_tokens: int
    max_duration_seconds: float
    max_cost_usd: float
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        self._started = monotonic()

    def check_duration(self) -> None:
        if monotonic() - self._started >= self.max_duration_seconds:
            raise WorkflowBudgetExceeded("task duration budget exceeded")

    def before_llm_call(self, messages: list[dict[str, Any]]) -> None:
        self.check_duration()
        context_chars = sum(len(str(item.get("content", ""))) for item in messages)
        if context_chars > self.max_context_chars:
            raise WorkflowBudgetExceeded("LLM context size budget exceeded")
        if self.calls >= self.max_calls:
            raise WorkflowBudgetExceeded("LLM call budget exceeded")
        self.calls += 1

    def record_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
    ) -> dict[str, Any]:
        prompt = int(prompt_tokens or 0)
        completion = int(completion_tokens or 0)
        total = int(total_tokens or (prompt + completion))
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.estimated_cost_usd += (
            prompt * self.input_cost_per_million
            + completion * self.output_cost_per_million
        ) / 1_000_000
        snapshot = self.snapshot()
        if self.total_tokens > self.max_total_tokens:
            raise WorkflowBudgetExceeded("LLM token budget exceeded")
        if self.max_cost_usd > 0 and self.estimated_cost_usd > self.max_cost_usd:
            raise WorkflowBudgetExceeded("LLM cost budget exceeded")
        self.check_duration()
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "elapsed_seconds": round(monotonic() - self._started, 3),
        }


def classify_failure(exc: BaseException) -> dict[str, Any]:
    """Classify failures without persisting exception messages or prompts."""
    name = type(exc).__name__.lower()
    module = type(exc).__module__.lower()
    if isinstance(exc, WorkflowBudgetExceeded):
        category, retryable = "budget", False
    elif "ratelimit" in name or "rate_limit" in name:
        category, retryable = "rate_limit", True
    elif "timeout" in name:
        category, retryable = "timeout", True
    elif any(token in name for token in ("connection", "network", "apierror")) or "httpx" in module:
        category, retryable = "network", True
    elif any(token in name for token in ("sql", "database", "operationalerror", "programmingerror")):
        category, retryable = "sql", False
    elif any(token in name for token in ("validation", "business", "unsupported")):
        category, retryable = "business", False
    else:
        category, retryable = "internal", False
    return {
        "category": category,
        "retryable": retryable,
        "error_type": type(exc).__name__,
    }
