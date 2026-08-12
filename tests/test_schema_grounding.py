import unittest

from agents.schema_grounding import (
    find_required_fields,
    invalid_qualified_columns,
    merge_required_fields,
    missing_required_columns,
    percentage_scale_issue,
)


class SchemaGroundingTests(unittest.TestCase):
    def test_membership_tier_is_required_for_explicit_business_term(self):
        required = find_required_fields("按会员等级统计客户数量")

        self.assertEqual(
            [(item["table_name"], item["column_name"]) for item in required],
            [("customers", "membership_tier")],
        )

    def test_required_field_survives_reranking(self):
        selected = [{
            "table_name": "customers",
            "column_name": "total_spend_usd",
            "relevance": "high",
        }]

        merged = merge_required_fields(selected, "分析会员等级")

        self.assertIn(
            ("customers", "membership_tier", "required"),
            [
                (item["table_name"], item["column_name"], item["relevance"])
                for item in merged
            ],
        )

    def test_spend_segmentation_cannot_replace_membership_tier(self):
        selected = merge_required_fields([], "按会员等级统计客户数量")
        wrong_sql = (
            "SELECT CASE WHEN total_spend_usd > 1000 THEN '高价值' END AS level, "
            "COUNT(*) FROM customers GROUP BY level"
        )
        correct_sql = (
            "SELECT membership_tier, COUNT(*) FROM customers GROUP BY membership_tier"
        )

        self.assertEqual(
            missing_required_columns(wrong_sql, selected),
            ["customers.membership_tier"],
        )
        self.assertEqual(missing_required_columns(correct_sql, selected), [])

    def test_qualified_column_must_belong_to_its_table_alias(self):
        wrong_sql = (
            "SELECT c.total_spend_usd, SUM(o.total_spend_usd) "
            "FROM customers c JOIN orders o ON c.customer_id=o.customer_id"
        )
        correct_sql = (
            "SELECT c.total_spend_usd, SUM(o.total_amount_usd) "
            "FROM customers c JOIN orders o ON c.customer_id=o.customer_id"
        )

        errors = invalid_qualified_columns(wrong_sql)
        self.assertTrue(any("o.total_spend_usd" in error for error in errors))
        self.assertEqual(invalid_qualified_columns(correct_sql), [])

    def test_percentage_query_rejects_fraction_scale(self):
        fraction_sql = "SELECT COUNT(*) / SUM(COUNT(*)) OVER() AS ratio FROM customers"
        percent_sql = "SELECT COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS pct FROM customers"

        self.assertIsNotNone(percentage_scale_issue("统计客户占比", fraction_sql))
        self.assertIsNone(percentage_scale_issue("统计客户占比", percent_sql))


if __name__ == "__main__":
    unittest.main()
