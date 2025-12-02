"""
Tiny demo to score a multi-turn plan trajectory with plan rewards.

Run:
  python scripts/demo_plan_rewards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from grpo_intent.plan_reward import (
    PlanRewardSettings,
    PlanStep,
    PlanTrajectory,
    compute_trajectory_reward,
)
from grpo_intent.plan_schema import PlanBranch, PlanNode, PlanSpec, ToolSpec


def simple_plan() -> PlanSpec:
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


def sample_trajectory() -> Trajectory:
    steps = [
        PlanStep(
            node_id="decide_refund",
            pred_slots={"order_id": "ORD123"},
            gold_slots={"order_id": "ORD123"},
            branch_taken="full",
            state_before={"has_account": True, "eligible_full": True, "refund_applied": False},
            state_after={"has_account": True, "eligible_full": True, "refund_applied": False},
            latency_ms=420,
            policy_ok=True,
        ),
        PlanStep(
            node_id="apply_refund",
            pred_slots={"amount": "100.00"},
            gold_slots={"amount": "100.00"},
            tool_call={"amount": 100.0},
            state_before={"has_account": True, "eligible_full": True, "refund_applied": False},
            state_after={"has_account": True, "eligible_full": True, "refund_applied": True},
            latency_ms=680,
            policy_ok=True,
        ),
    ]
    return PlanTrajectory(steps=steps, plan_completed=True)


def main():
    plan = simple_plan()
    traj = sample_trajectory()
    cfg = PlanRewardSettings()
    reward = compute_trajectory_reward(traj, plan, cfg)
    print("Per-step rewards:", reward.per_step)
    print("Discounted total:", reward.discounted_total)
    print("Completion bonus:", reward.done_bonus)


if __name__ == "__main__":
    main()
