import unittest

from evaluation.metrics.text2sql_metrics import compare_result_sets


class Text2SqlMetricTests(unittest.TestCase):
    def test_equivalent_results_allow_different_aliases(self):
        generated = [{"tier": "Gold", "customer_count": 10, "ratio": 25.0}]
        expected = [{"membership_tier": "Gold", "cnt": 10, "pct": 25.0}]

        self.assertTrue(compare_result_sets(generated, expected))

    def test_different_percentage_scales_are_not_equivalent(self):
        generated = [{"tier": "Gold", "count": 10, "ratio": 0.25}]
        expected = [{"membership_tier": "Gold", "cnt": 10, "pct": 25.0}]

        self.assertFalse(compare_result_sets(generated, expected))


if __name__ == "__main__":
    unittest.main()
