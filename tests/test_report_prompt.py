import unittest

from config.prompts.report_prompt import build_report_prompt


class ReportPromptTests(unittest.TestCase):
    def test_prompt_contains_traceable_sql_evidence(self):
        prompt = build_report_prompt(
            user_query="分析会员等级",
            analysis_result={},
            prediction_result={},
            governance_result={},
            evidence={
                "sql": "SELECT membership_tier, COUNT(*) FROM customers GROUP BY membership_tier",
                "columns": ["membership_tier", "count"],
                "rows": [{"membership_tier": "Gold", "count": 10}],
                "row_count": 1,
                "task_plan": {"analysis_tools": ["query_summary"]},
            },
        )

        self.assertIn("唯一允许引用的事实来源", prompt)
        self.assertIn("SELECT membership_tier", prompt)
        self.assertIn('"Gold"', prompt)
        self.assertIn("不得补造数字", prompt)


if __name__ == "__main__":
    unittest.main()
