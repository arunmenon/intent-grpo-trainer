import unittest

from grpo_intent.router_reward import RouterRewardSettings, RouterSimulator
from grpo_intent.router_schema import RouterAction, RouterActionType, RouterObservation, RouterTool


class RouterRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = [
            RouterTool(
                tool_id="SmallLLM",
                kind="LLM",
                estimated_latency_ms_p50=200,
                estimated_cost_usd_per_call=0.001,
                reliability=0.99,
                domain_tags=[],
                schema_in={},
                schema_out={},
            ),
            RouterTool(
                tool_id="BigLLM",
                kind="LLM",
                estimated_latency_ms_p50=800,
                estimated_cost_usd_per_call=0.01,
                reliability=0.97,
                domain_tags=[],
                schema_in={},
                schema_out={},
            ),
        ]
        self.obs = RouterObservation(
            request_id="r1",
            turn_id=0,
            query="test",
            conversation_history=[],
            tool_menu=self.tools,
        )

    def test_big_model_penalty(self):
        settings = RouterRewardSettings(big_model_latency_ms=400, big_model_cost_usd=0.002)
        sim = RouterSimulator(settings=settings)
        actions = [
            RouterAction(decision_id="a0", step_index=0, action_type=RouterActionType.CALL_TOOL, tool_id="SmallLLM"),
            RouterAction(decision_id="a1", step_index=1, action_type=RouterActionType.CALL_TOOL, tool_id="BigLLM"),
            RouterAction(decision_id="a2", step_index=2, action_type=RouterActionType.ANSWER, answer="done"),
        ]
        metrics = sim.run_episode(self.obs, actions)
        self.assertEqual(metrics.big_model_calls, 1)
        # Since reward is clipped to [-5, 5], ensure we get a finite value and success is counted.
        self.assertEqual(metrics.success, 1)
        self.assertTrue(-5.0 <= metrics.reward <= 5.0)


if __name__ == "__main__":
    unittest.main()
