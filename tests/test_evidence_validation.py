from agents.evidence import build_evidence_bundle
from agents.evidence_validation import validate_query_evidence, validate_report_claims


def _request(**updates):
    request = {
        "schema_version": "1.0",
        "metric_catalog_version": "2026.09.1",
        "metrics": ["paid_amount"],
        "dimensions": ["country"],
        "time_scope": {"field": "orders.order_date", "years": [2025]},
        "filters": {"country": ["United States"]},
        "sort": [{"field": "paid_amount", "direction": "desc"}],
        "limit": 2,
        "data_source": "ai_analytics",
    }
    request.update(updates)
    return request


def test_query_evidence_verifies_metric_filters_sort_and_limit():
    validation = validate_query_evidence(
        sql=(
            "SELECT c.country, SUM(o.total_amount_usd) AS paid_amount "
            "FROM orders o JOIN customers c ON c.customer_id=o.customer_id "
            "WHERE c.country='United States' AND YEAR(o.order_date)=2025 "
            "GROUP BY c.country ORDER BY paid_amount DESC LIMIT 2"
        ),
        analysis_request=_request(),
        rows=[{"country": "United States", "paid_amount": 120.0}],
    )
    assert validation["status"] == "verified"
    assert {"orders", "customers"} == set(validation["source_tables"])


def test_query_evidence_rejects_lost_conditions_and_wrong_order():
    validation = validate_query_evidence(
        sql=(
            "SELECT c.country, SUM(o.total_amount_usd) AS paid_amount "
            "FROM orders o JOIN customers c ON c.customer_id=o.customer_id "
            "GROUP BY c.country ORDER BY paid_amount ASC LIMIT 2"
        ),
        analysis_request=_request(),
        rows=[
            {"country": "China", "paid_amount": 200.0},
            {"country": "United States", "paid_amount": 100.0},
        ],
    )
    failed = {item["code"] for item in validation["checks"] if item["status"] == "failed"}
    assert validation["status"] == "failed"
    assert {"filter.country", "filter.year", "result.sort"}.issubset(failed)


def test_query_evidence_detects_join_fanout_for_customer_count():
    request = _request(
        metrics=["customer_count"], dimensions=[], time_scope={"years": []},
        filters={}, sort=[], limit=None,
    )
    unsafe = validate_query_evidence(
        sql=(
            "SELECT COUNT(*) AS customer_count FROM customers c "
            "JOIN orders o ON o.customer_id=c.customer_id"
        ),
        analysis_request=request,
        rows=[{"customer_count": 10}],
    )
    assert any(
        item["code"] == "join.fanout" and item["status"] == "failed"
        for item in unsafe["checks"]
    )
    safe = validate_query_evidence(
        sql=(
            "SELECT COUNT(DISTINCT c.customer_id) AS customer_count FROM customers c "
            "JOIN orders o ON o.customer_id=c.customer_id"
        ),
        analysis_request=request,
        rows=[{"customer_count": 4}],
    )
    assert safe["status"] == "verified"

    duplicate_names = validate_query_evidence(
        sql=(
            "SELECT COUNT(DISTINCT c.name) AS customer_count FROM customers c "
            "JOIN orders o ON o.customer_id=c.customer_id"
        ),
        analysis_request=request,
        rows=[{"customer_count": 2}],
    )
    assert any(
        item["code"] == "join.fanout" and item["status"] == "failed"
        for item in duplicate_names["checks"]
    )


def test_empty_and_null_results_are_not_falsely_verified():
    sql = "SELECT AVG(resolution_hours) AS avg_resolution_hours FROM support_tickets"
    request = _request(
        metrics=["avg_resolution_hours"], dimensions=[], time_scope={"years": []},
        filters={}, sort=[], limit=None, data_source="support_ops",
    )
    assert validate_query_evidence(sql=sql, analysis_request=request, rows=[])["status"] == "unverifiable"
    assert validate_query_evidence(
        sql=sql, analysis_request=request, rows=[{"avg_resolution_hours": None}]
    )["status"] == "unverifiable"


def test_aggregate_grain_and_metric_result_column_are_required():
    no_group = validate_query_evidence(
        sql="SELECT SUM(total_amount_usd) AS paid_amount FROM orders",
        analysis_request=_request(
            filters={}, time_scope={"years": []}, sort=[], limit=None
        ),
        rows=[{"country": "United States", "paid_amount": 50.0}],
    )
    assert any(
        item["code"] == "grain.country" and item["status"] == "failed"
        for item in no_group["checks"]
    )
    opaque_output = validate_query_evidence(
        sql="SELECT SUM(total_amount_usd) AS value FROM orders",
        analysis_request=_request(
            dimensions=[], filters={}, time_scope={"years": []}, sort=[], limit=None
        ),
        rows=[{"value": 50.0}],
    )
    assert any(
        item["code"] == "result.metric.paid_amount" and item["status"] == "failed"
        for item in opaque_output["checks"]
    )


def test_unavailable_refund_metric_fails_closed():
    validation = validate_query_evidence(
        sql="SELECT SUM(total_amount_usd) AS net_revenue FROM orders",
        analysis_request=_request(
            metrics=["net_revenue"], dimensions=[], filters={},
            time_scope={"years": []}, sort=[], limit=None,
        ),
        rows=[{"net_revenue": 50.0}],
    )
    assert validation["status"] == "failed"
    assert any(item["code"] == "metric.net_revenue" for item in validation["checks"])


def test_facts_keep_numeric_period_as_dimension_and_attach_provenance():
    evidence = build_evidence_bundle(
        user_query="2025年每月营收",
        task_plan={
            "analysis_request": _request(
                metrics=["revenue"], dimensions=["time"],
                time_scope={"years": [2025]}, filters={}, sort=[], limit=None,
            )
        },
        sql=(
            "SELECT year, month, SUM(revenue_usd) AS revenue_usd "
            "FROM monthly_revenue WHERE year=2025 GROUP BY year, month"
        ),
        rows=[{"year": 2025, "month": 1, "revenue_usd": 88.5}],
    )
    assert [fact["label"] for fact in evidence["facts"]] == ["revenue_usd"]
    assert evidence["facts"][0]["dimensions"] == {"year": 2025, "month": 1}
    assert evidence["facts"][0]["metric_catalog_version"] == "2026.09.1"
    assert evidence["facts"][0]["unit"] == "USD"


def test_report_claims_verify_values_dimensions_periods_and_rankings():
    evidence = {
        "analysis_request": {"time_scope": {"years": [2025]}},
        "facts": [
            {"fact_id": "F1", "value": 120.0, "dimensions": {"country": "United States"}},
            {"fact_id": "F2", "value": 80.0, "dimensions": {"country": "China"}},
        ],
    }
    valid = validate_report_claims(
        "2025年 United States 最高，为 120 USD [F1]。", evidence
    )
    assert valid["status"] == "verified"

    assert validate_report_claims("United States 为 999 USD [F1]。", evidence)["status"] == "failed"
    assert validate_report_claims("United States 为 120，而目标为 999 USD [F1]。", evidence)["status"] == "failed"
    assert validate_report_claims("United States 为 -120 USD [F1]。", evidence)["status"] == "failed"
    assert validate_report_claims("2024年 United States 为 120 USD [F1]。", evidence)["status"] == "failed"
    assert validate_report_claims("China 为 120 USD [F1]。", evidence)["status"] == "failed"
    assert validate_report_claims("United States 为 120 USD。", evidence)["status"] == "failed"
