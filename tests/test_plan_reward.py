import unittest

from grpo_intent.plan_reward import (
    PlanRewardResult,
    PlanRewardSettings,
    PlanStep,
    PlanTrajectory,
    compute_step_reward,
    compute_trajectory_reward,
)
from grpo_intent.plan_schema import PlanBranch, PlanNode, PlanSpec, ToolSpec


def _simple_plan():
    def has_account(state):
        return state.get("has_account", False)

    def refund_effect(before, after):
        return bool(after.get("refund_applied") and not before.get("refund_applied"))

    def full_refund_guard(state):
        return bool(state.get("eligible_full"))

    def partial_refund_guard(state):
        return not state.get("eligible_full", False)

    def tool_validator(call):
        return float(call.get("amount", 0) > 0)

    decide = PlanNode(
        node_id="decide_refund",
        required_slots=["order_id"],
        preconditions=[has_account],
        branches=[
            PlanBranch(branch_id="full", guard=full_refund_guard),
            PlanBranch(branch_id="partial", guard=partial_refund_guard),
        ],
    )

    refund = PlanNode(
        node_id="apply_refund",
        required_slots=["amount"],
        preconditions=[has_account],
        effects=[refund_effect],
        tool_spec=ToolSpec(name="refund_api", validator=tool_validator),
    )

    return PlanSpec(nodes={decide.node_id: decide, refund.node_id: refund}, start_node=decide.node_id)


class PlanRewardTests(unittest.TestCase):
    def test_step_reward_weights_required_slots(self):
        plan = _simple_plan()
        settings = PlanRewardSettings(required_slot_weight=1.0, optional_slot_weight=0.25)

        step_missing_required = PlanStep(
            node_id="decide_refund",
            pred_slots={},
            gold_slots={"order_id": "ORD1"},
            branch_taken="full",
            state_before={"has_account": True, "eligible_full": True},
            state_after={"has_account": True, "eligible_full": True},
        )
        step_with_required = PlanStep(
            node_id="decide_refund",
            pred_slots={"order_id": "ORD1"},
            gold_slots={"order_id": "ORD1"},
            branch_taken="full",
            state_before={"has_account": True, "eligible_full": True},
            state_after={"has_account": True, "eligible_full": True},
        )

        reward_missing = compute_step_reward(step_missing_required, plan, settings)
        reward_present = compute_step_reward(step_with_required, plan, settings)

        self.assertLess(reward_missing, reward_present)

    def test_trajectory_reward_adds_completion_bonus(self):
        plan = _simple_plan()
        settings = PlanRewardSettings(w_done=1.0, gamma=0.9, discount_done=True)
        steps = [
            PlanStep(
                node_id="decide_refund",
                pred_slots={"order_id": "ORD1"},
                gold_slots={"order_id": "ORD1"},
                branch_taken="full",
                state_before={"has_account": True, "eligible_full": True, "refund_applied": False},
                state_after={"has_account": True, "eligible_full": True, "refund_applied": False},
            ),
            PlanStep(
                node_id="apply_refund",
                pred_slots={"amount": "50"},
                gold_slots={"amount": "50"},
                tool_call={"amount": 50},
                state_before={"has_account": True, "eligible_full": True, "refund_applied": False},
                state_after={"has_account": True, "eligible_full": True, "refund_applied": True},
            ),
        ]
        traj = PlanTrajectory(steps=steps, plan_completed=True)
        reward: PlanRewardResult = compute_trajectory_reward(traj, plan, settings)

        base_total = sum(reward.per_step[i] * (settings.gamma ** i) for i in range(len(reward.per_step)))
        expected_bonus = settings.w_done * (settings.gamma ** len(steps))
        self.assertAlmostEqual(reward.done_bonus, expected_bonus, places=6)
        self.assertAlmostEqual(reward.discounted_total, base_total + expected_bonus, places=6)


if __name__ == "__main__":
    unittest.main()
