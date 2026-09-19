import unittest
import json

from agents.analysis_request import CONFIRMED_REQUEST_PREFIX, build_analysis_request
from agents.task_planning import (
    build_task_plan,
    deterministic_evidence_sql,
    planned_nodes,
)


class TaskPlanningTests(unittest.TestCase):
    def test_category_return_analysis_only_uses_query_summary(self):
        plan = build_task_plan("分析各品类退货率", "analysis")

        self.assertEqual(plan["metrics"], ["return_rate"])
        self.assertIn("category", plan["dimensions"])
        self.assertEqual(plan["analysis_tools"], ["query_summary"])
        self.assertEqual(plan["prediction_tools"], [])

    def test_sales_forecast_does_not_run_churn_model(self):
        plan = build_task_plan("预测未来6个月销售趋势", "prediction")

        self.assertEqual(plan["prediction_tools"], ["sales_forecast"])
        self.assertEqual(plan["prediction_horizon_months"], 6)
        self.assertNotIn("Analysis Agent", planned_nodes(plan))
        self.assertIn("Prediction Agent", planned_nodes(plan))

    def test_next_year_forecast_uses_twelve_month_horizon(self):
        plan = build_task_plan("预测下一年各月营收", "prediction")

        self.assertEqual(plan["prediction_horizon_months"], 12)

    def test_rfm_churn_report_selects_only_requested_specialists(self):
        plan = build_task_plan(
            "对客户做RFM分析和流失预测，生成综合经营报告", "mixed"
        )

        self.assertIn("rfm", plan["analysis_tools"])
        self.assertIn("churn_prediction", plan["prediction_tools"])
        self.assertNotIn("sales_forecast", plan["prediction_tools"])
        self.assertTrue(plan["need_report"])

    def test_plain_report_does_not_force_prediction(self):
        plan = build_task_plan("生成会员等级经营报告", "mixed")

        self.assertEqual(plan["prediction_tools"], [])
        self.assertEqual(plan["analysis_tools"], ["query_summary"])
        self.assertIn("Report Agent", planned_nodes(plan))

    def test_rfm_uses_reproducible_customer_evidence_query(self):
        plan = build_task_plan("进行客户RFM分析", "analysis")

        sql = deterministic_evidence_sql(plan)

        self.assertIn("FROM customers", sql)
        self.assertNotIn("JOIN orders", sql)

    def test_sales_forecast_uses_monthly_history_contract(self):
        plan = build_task_plan("预测未来6个月销售趋势", "prediction")

        sql = deterministic_evidence_sql(plan)

        self.assertIn("FROM monthly_revenue", sql)
        self.assertIn("year, month", sql)

    def test_static_customer_metric_rejects_unsupported_year_scope(self):
        plan = build_task_plan("2025年美国客户RFM分层", "analysis")
        self.assertIsNone(deterministic_evidence_sql(plan))

    def test_confirmed_multi_turn_contract_is_not_reparsed_away(self):
        request = build_analysis_request("统计2025年各地区实付金额")
        request["dimensions"] = ["time"]
        request["filters"] = {"country": ["United States"]}
        query = (
            CONFIRMED_REQUEST_PREFIX
            + json.dumps(request, ensure_ascii=False)
            + "\n当前用户表达：按月拆分"
        )

        plan = build_task_plan(query, "analysis")

        self.assertEqual(plan["analysis_request"], request)
        self.assertIn("time", plan["dimensions"])
        self.assertEqual(plan["filters"]["years"], [2025])
        self.assertEqual(plan["filters"]["countries"], ["United States"])


if __name__ == "__main__":
    unittest.main()
