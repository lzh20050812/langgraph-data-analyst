import unittest

from agents.state import create_initial_state


class InitialStateTests(unittest.TestCase):
    def test_explicit_intent_and_evidence_contract_are_initialized(self):
        state = create_initial_state("预测未来销售", requested_intent="prediction")

        self.assertEqual(state["requested_intent"], "prediction")
        self.assertEqual(state["task_plan"], {})
        self.assertEqual(state["evidence"], {})


if __name__ == "__main__":
    unittest.main()
