"""Strongly typed contracts shared across workflow nodes.

The public workflow state remains dictionary-compatible for LangGraph and for
the existing evaluation suite.  These Pydantic models validate the important
cross-agent payloads at their creation boundaries so malformed plans or
evidence fail early instead of surfacing several nodes later.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Intent = Literal["sql_query", "analysis", "prediction", "mixed"]


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    analysis_tools: list[str] = Field(default_factory=list)
    prediction_tools: list[str] = Field(default_factory=list)
    prediction_horizon_months: int | None = Field(default=None, ge=1, le=120)
    need_report: bool = False
    analysis_request: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tool_intent(self) -> "TaskPlan":
        if self.intent == "sql_query" and (
            self.analysis_tools or self.prediction_tools or self.need_report
        ):
            raise ValueError("sql_query plans cannot request specialist tools or reports")
        return self


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_query: str = Field(min_length=1)
    task_plan: dict[str, Any]
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(ge=0)
    analysis_request: dict[str, Any] = Field(default_factory=dict)
    metric_catalog_version: str | None = None
    source_tables: list[str] = Field(default_factory=list)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    query_contract: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    report_validation: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    analysis_results: dict[str, Any] = Field(default_factory=dict)
    prediction_results: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_row_count(self) -> "EvidenceBundle":
        if self.row_count != len(self.rows):
            raise ValueError("evidence row_count must match the materialized rows")
        return self


class NodeTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str
    status: Literal["completed", "failed"]
    duration_ms: float = Field(ge=0)
    current_step: str = ""
    error: str | None = None


def validate_task_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return TaskPlan.model_validate(plan).model_dump(mode="python")


def validate_evidence(bundle: dict[str, Any]) -> dict[str, Any]:
    return EvidenceBundle.model_validate(bundle).model_dump(mode="python")
