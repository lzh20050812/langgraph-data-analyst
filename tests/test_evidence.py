import unittest

from agents.evidence import build_evidence_bundle, update_evidence


class EvidenceBundleTests(unittest.TestCase):
    def test_bundle_preserves_sql_rows_and_plan(self):
        rows = [{"category": "Books", "return_rate": 0.1}]
        plan = {"analysis_tools": ["query_summary"]}

        evidence = build_evidence_bundle(
            user_query="分析退货率",
            task_plan=plan,
            sql="SELECT category, return_rate FROM product_summary",
            rows=rows,
        )

        self.assertEqual(evidence["columns"], ["category", "return_rate"])
        self.assertEqual(evidence["row_count"], 1)
        self.assertEqual(evidence["rows"], rows)
        self.assertEqual(evidence["task_plan"], plan)

    def test_nodes_update_one_shared_bundle(self):
        evidence = build_evidence_bundle(
            user_query="query", task_plan={}, sql="SELECT 1", rows=[{"v": 1}]
        )

        updated = update_evidence(evidence, data_quality={"score": 100})
        updated = update_evidence(updated, analysis_results={"row_count": 1})

        self.assertIs(updated, evidence)
        self.assertEqual(updated["data_quality"]["score"], 100)
        self.assertEqual(updated["analysis_results"]["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
