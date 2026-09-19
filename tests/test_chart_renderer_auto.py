from agents.chart_renderer import build_query_result_chart, render_charts
from agents.planner import route_after_governance
from agents.state import create_initial_state


def test_sql_result_with_time_and_measure_generates_line_chart():
    chart = build_query_result_chart([
        {"year": 2025, "month": 1, "revenue_usd": 100.0},
        {"year": 2025, "month": 2, "revenue_usd": 120.0},
    ])

    assert chart["type"] == "line"
    assert chart["option"]["xAxis"]["data"] == ["2025-01", "2025-02"]
    assert chart["option"]["series"][0]["name"] == "revenue_usd"


def test_sql_result_with_dimension_and_measure_generates_bar_chart():
    chart = build_query_result_chart([
        {"membership_tier": "Gold", "customer_count": 42},
        {"membership_tier": "Silver", "customer_count": 31},
    ])

    assert chart["type"] == "bar"
    assert chart["option"]["xAxis"]["data"] == ["Gold", "Silver"]
    assert chart["generated_by"] == "query_result_auto"


def test_renderer_only_adds_generic_chart_when_specialist_charts_are_absent():
    state = create_initial_state("统计会员等级客户数")
    state["query_result"] = [{"membership_tier": "Gold", "count": 42}]

    rendered = render_charts(state)

    assert [chart["id"] for chart in rendered["charts"]] == ["query_result_auto"]
    assert route_after_governance(rendered) == "chart_renderer"
