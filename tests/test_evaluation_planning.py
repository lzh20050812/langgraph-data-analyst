import unittest

from evaluation.metrics.multi_agent_metrics_v4 import compute_full_pipeline_rate


class EvaluationPlanningTests(unittest.TestCase):
    def test_on_demand_plan_does_not_require_unrequested_prediction(self):
        details = [{
            "id": "q1",
            "task_type": "mixed",
            "planned_agents": [
                "Planner",
                "Schema Agent",
                "SQL Agent",
                "Governance Agent",
                "Analysis Agent",
                "Report Agent",
                "Chart Renderer",
            ],
            "agents_invoked": [
                "Planner",
                "Schema Agent",
                "SQL Agent",
                "Governance Agent",
                "Analysis Agent",
                "Report Agent",
                "Chart Renderer",
            ],
            "agent_trace": [],
        }]

        result = compute_full_pipeline_rate(details)

        self.assertEqual(result["overall_completion_rate"], 1.0)
        self.assertNotIn(
            "Prediction Agent", result["per_task"][0]["planned"]
        )


if __name__ == "__main__":
    unittest.main()
