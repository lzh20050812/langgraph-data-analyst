"""Versioned business knowledge kept separate from physical schema metadata."""

from __future__ import annotations

from typing import Any

from agents.analysis_request import METRIC_CATALOG, METRIC_CATALOG_VERSION
from agents.schema_retrieval import _term_score


BUSINESS_CATALOG_VERSION = METRIC_CATALOG_VERSION
BUSINESS_SOURCE_ID = "repo:agents/analysis_request.py"


def business_knowledge_entries() -> list[dict[str, Any]]:
    entries = []
    for metric in METRIC_CATALOG.values():
        data_source = (
            "support_ops"
            if metric.source_table.startswith("support_")
            else "ai_analytics"
        )
        entries.append({
            "knowledge_type": "metric",
            "knowledge_id": f"metric:{metric.metric_id}",
            "data_source": data_source,
            "version": metric.version,
            "catalog_version": BUSINESS_CATALOG_VERSION,
            "source_id": BUSINESS_SOURCE_ID,
            "access_scope": "public",
            "owner_id": "",
            "name": metric.name,
            "aliases": metric.aliases,
            "definition": metric.formula,
            "required_fields": metric.required_fields,
            "supported_dimensions": metric.supported_dimensions,
            "limitations": metric.limitations,
        })
    return entries


def search_business_knowledge(
    query: str,
    *,
    data_source: str = "ai_analytics",
    principal_id: str | None = None,
    top_k: int = 5,
    entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return only business definitions visible in the requested data source."""
    ranked = []
    for entry in entries or business_knowledge_entries():
        if entry.get("data_source") != data_source:
            continue
        if entry.get("access_scope", "public") != "public" and (
            not principal_id or entry.get("owner_id") != principal_id
        ):
            continue
        terms = [entry.get("name", ""), *entry.get("aliases", [])]
        score = max((_term_score(query, term) for term in terms), default=0.0)
        if score > 0:
            ranked.append((score, entry))
    ranked.sort(key=lambda item: (-item[0], item[1]["knowledge_id"]))
    return [dict(entry, score=round(score, 6)) for score, entry in ranked[:top_k]]


def format_business_context(entries: list[dict[str, Any]]) -> str:
    lines = []
    for entry in entries:
        limits = "；".join(entry.get("limitations") or []) or "无额外限制"
        lines.append(
            f"- {entry['name']} [{entry['knowledge_id']}@{entry['version']}]: "
            f"{entry['definition']}；必要字段={','.join(entry['required_fields'])}；"
            f"限制={limits}；来源={entry['source_id']}"
        )
    return "\n".join(lines)
