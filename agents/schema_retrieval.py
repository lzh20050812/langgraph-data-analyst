"""Hybrid lexical and dense retrieval for database schema fields."""

from __future__ import annotations

import re
from typing import Iterable

from storage.chromadb.schema_metadata import (
    SCHEMA_FIELDS,
    SCHEMA_RELATIONSHIPS,
    get_schema_fields,
    is_accessible,
)


_NON_SEARCHABLE = re.compile(r"[^0-9a-z_\u4e00-\u9fff]+")
_ASCII_TOKEN = re.compile(r"[a-z][a-z0-9_]*")


def _normalize(text: object) -> str:
    return _NON_SEARCHABLE.sub("", str(text or "").lower())


def _chinese_ngrams(text: str, size: int = 2) -> set[str]:
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    if not chinese:
        return set()
    if len(chinese) < size:
        return {chinese}
    return {chinese[index:index + size] for index in range(len(chinese) - size + 1)}


def _term_score(query: str, term: object) -> float:
    """Score one schema term without requiring an external tokenizer."""
    normalized_query = _normalize(query)
    normalized_term = _normalize(term)
    if not normalized_query or not normalized_term:
        return 0.0

    score = 0.0
    if len(normalized_term) >= 2 and normalized_term in normalized_query:
        score += 6.0
    elif len(normalized_query) >= 2 and normalized_query in normalized_term:
        score += 2.0

    query_ascii = set(_ASCII_TOKEN.findall(normalized_query))
    term_ascii = set(_ASCII_TOKEN.findall(normalized_term))
    if term_ascii:
        score += 2.0 * len(query_ascii & term_ascii) / len(term_ascii)

    term_ngrams = _chinese_ngrams(normalized_term)
    if term_ngrams:
        query_ngrams = _chinese_ngrams(normalized_query)
        score += 2.0 * len(query_ngrams & term_ngrams) / len(term_ngrams)
    return score


def lexical_schema_search(
    query: str,
    top_k: int = 20,
    *,
    data_source: str = "ai_analytics",
    principal_id: str | None = None,
    schema_fields: list[dict] | None = None,
) -> list[dict]:
    """Rank schema fields by names, business terms and curated aliases."""
    ranked = []
    fields = (
        get_schema_fields(data_source, principal_id)
        if schema_fields is None
        else [
            field for field in schema_fields
            if field.get("data_source", "ai_analytics") == data_source
            and is_accessible(field, principal_id)
        ]
    )
    for field in fields:
        weighted_terms = [
            (field["column_name"], 1.15),
            (field.get("business_term", ""), 1.1),
            *((alias, 1.0) for alias in field.get("aliases", [])),
            (field["table_name"], 0.35),
        ]
        term_scores = [
            _term_score(query, term) * weight
            for term, weight in weighted_terms
        ]
        score = max(term_scores, default=0.0)
        # Multiple matching aliases are useful evidence, but should not swamp an
        # exact field-name match.
        score += 0.08 * sum(value for value in term_scores if value > 0)
        if score <= 0:
            continue
        ranked.append((score, field))

    ranked.sort(
        key=lambda item: (
            -item[0],
            item[1]["table_name"],
            item[1]["column_name"],
        )
    )
    results = []
    for rank, (score, field) in enumerate(ranked[:max(1, top_k)], start=1):
        results.append({
            "rank": rank,
            "table_name": field["table_name"],
            "column_name": field["column_name"],
            "business_term": field.get("business_term", ""),
            "dtype": field.get("dtype", ""),
            "data_source": field.get("data_source", data_source),
            "catalog_version": field.get("catalog_version", ""),
            "source_id": field.get("source_id", ""),
            "access_scope": field.get("access_scope", "public"),
            "document": " | ".join([
                field["table_name"],
                field["column_name"],
                field.get("business_term", ""),
                ", ".join(field.get("aliases", [])),
            ]),
            "score": round(float(score), 6),
        })
    return results


def fuse_schema_candidates(
    vector_candidates: Iterable[dict],
    lexical_candidates: Iterable[dict],
    top_k: int = 15,
    vector_weight: float = 0.2,
    lexical_weight: float = 0.8,
    rrf_k: int = 20,
) -> list[dict]:
    """Fuse two rankings with weighted reciprocal-rank fusion."""
    if vector_weight < 0 or lexical_weight < 0:
        raise ValueError("retrieval weights must be non-negative")
    if vector_weight + lexical_weight <= 0:
        raise ValueError("at least one retrieval weight must be positive")

    fused: dict[tuple[str, str], dict] = {}
    rankings = (
        ("vector", list(vector_candidates), vector_weight),
        ("lexical", list(lexical_candidates), lexical_weight),
    )
    for source, candidates, weight in rankings:
        for fallback_rank, candidate in enumerate(candidates, start=1):
            key = (candidate["table_name"], candidate["column_name"])
            rank = max(1, int(candidate.get("rank") or fallback_rank))
            item = fused.setdefault(key, {
                "table_name": key[0],
                "column_name": key[1],
                "business_term": candidate.get("business_term", ""),
                "dtype": candidate.get("dtype", ""),
                "document": candidate.get("document", ""),
                "hybrid_score": 0.0,
                "retrieval_sources": [],
            })
            item["hybrid_score"] += weight / (rrf_k + rank)
            item[f"{source}_rank"] = rank
            item[f"{source}_score"] = candidate.get("score")
            item["retrieval_sources"].append(source)
            for field_name in (
                "business_term", "dtype", "document", "data_source",
                "catalog_version", "source_id", "access_scope",
            ):
                if not item.get(field_name) and candidate.get(field_name):
                    item[field_name] = candidate[field_name]

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -item["hybrid_score"],
            item["table_name"],
            item["column_name"],
        ),
    )[:max(1, top_k)]
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
        item["score"] = round(float(item.pop("hybrid_score")), 8)
        item["retrieval_sources"] = sorted(set(item["retrieval_sources"]))
    return ordered


def hybrid_schema_search(
    query: str,
    vector_candidates: Iterable[dict],
    top_k: int = 15,
    *,
    data_source: str = "ai_analytics",
    principal_id: str | None = None,
) -> list[dict]:
    """Run lexical retrieval and fuse it with supplied dense candidates."""
    lexical_candidates = lexical_schema_search(
        query, top_k=max(top_k, 20), data_source=data_source,
        principal_id=principal_id,
    )
    return fuse_schema_candidates(
        vector_candidates=vector_candidates,
        lexical_candidates=lexical_candidates,
        top_k=top_k,
    )


def complete_relationship_paths(
    selected_fields: Iterable[dict],
    *,
    data_source: str = "ai_analytics",
    principal_id: str | None = None,
) -> list[dict]:
    """Add both join keys when retrieved tables require a known relationship."""
    result = [dict(item) for item in selected_fields]
    tables = {item.get("table_name") for item in result}
    existing = {(item.get("table_name"), item.get("column_name")) for item in result}
    field_map = {
        (field["table_name"], field["column_name"]): field
        for field in get_schema_fields(data_source, principal_id)
    }
    for relationship in SCHEMA_RELATIONSHIPS:
        if relationship.get("data_source") != data_source or not is_accessible(
            relationship, principal_id
        ):
            continue
        left = relationship["left_table"]
        right = relationship["right_table"]
        if not {left, right}.issubset(tables):
            continue
        for key in (
            (left, relationship["left_column"]),
            (right, relationship["right_column"]),
        ):
            if key in existing or key not in field_map:
                continue
            field = field_map[key]
            result.append({
                "table_name": key[0], "column_name": key[1],
                "dtype": field.get("dtype", ""),
                "business_term": field.get("business_term", ""),
                "aliases": field.get("aliases", []), "relevance": "required_join",
                "data_source": data_source,
                "catalog_version": field.get("catalog_version", ""),
                "source_id": relationship.get("source_id", ""),
                "relationship_version": relationship.get("version", ""),
            })
            existing.add(key)
    return result
