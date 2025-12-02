"""
Plan-aware reward runner (multi-turn) using the plan reward helpers.

This script computes per-step and discounted trajectory rewards for plan-based
rollouts. It mirrors the single-turn runner in spirit but stays model-agnostic:
you bring trajectories, it scores them.

Usage (built-in demo data and plan):
  python scripts/run_plan_grpo.py --demo

Usage (JSONL trajectories):
  python scripts/run_plan_grpo.py --dataset /path/to/plan_trajectories.jsonl

Expected dataset format (JSONL):
{
  "plan_completed": true,
  "steps": [
    {
      "node_id": "decide_refund",
      "pred_slots": {"order_id": "ORD123"},
      "gold_slots": {"order_id": "ORD123"},
      "branch_taken": "full",
      "tool_call": null,
      "state_before": {"has_account": true, "eligible_full": true, "refund_applied": false},
      "state_after": {"has_account": true, "eligible_full": true, "refund_applied": false},
      "latency_ms": 420,
      "policy_ok": true
    }
  ]
}

Plan spec: for now, we use a built-in refund plan. If you need custom plans,
adapt the `build_plan` function to construct PlanSpec from your metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from grpo_intent.plan_reward import (
    PlanRewardResult,
    PlanRewardSettings,
    PlanStep,
    PlanTrajectory,
    compute_trajectory_reward,
)
from grpo_intent.plan_schema import PlanBranch, PlanNode, PlanSpec, ToolSpec


def build_plan() -> PlanSpec:
    """Simple two-node refund plan with branching guards and a tool validator."""

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


def demo_trajectory() -> PlanTrajectory:
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


def load_dataset(path: Path) -> List[PlanTrajectory]:
    trajectories: List[PlanTrajectory] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            steps: List[PlanStep] = []
            for raw_step in record.get("steps", []):
                steps.append(
                    PlanStep(
                        node_id=raw_step["node_id"],
                        pred_slots=raw_step.get("pred_slots", {}),
                        gold_slots=raw_step.get("gold_slots", {}),
                        branch_taken=raw_step.get("branch_taken"),
                        tool_call=raw_step.get("tool_call"),
                        state_before=raw_step.get("state_before", {}),
                        state_after=raw_step.get("state_after", {}),
                        latency_ms=raw_step.get("latency_ms", 0.0),
                        policy_ok=bool(raw_step.get("policy_ok", True)),
                        t=raw_step.get("t"),
                    )
                )
            trajectories.append(PlanTrajectory(steps=steps, plan_completed=bool(record.get("plan_completed", False))))
    if not trajectories:
        raise SystemExit(f"No trajectories loaded from {path}")
    return trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="Score plan-based trajectories with plan rewards.")
    parser.add_argument("--dataset", type=Path, help="JSONL of trajectories (see docstring for schema).")
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo trajectory.")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor.")
    parser.add_argument("--tau-ms", type=float, default=1500.0, help="Latency shaping constant.")
    return parser.parse_args()


def main():
    args = parse_args()
    plan = build_plan()
    settings = PlanRewardSettings(gamma=args.gamma, tau_ms=args.tau_ms)

    if args.demo or not args.dataset:
        trajectories = [demo_trajectory()]
    else:
        if not args.dataset.exists():
            raise SystemExit(f"Dataset not found: {args.dataset}")
        trajectories = load_dataset(args.dataset)

    totals = []
    for idx, traj in enumerate(trajectories):
        reward: PlanRewardResult = compute_trajectory_reward(traj, plan, settings)
        totals.append(reward.discounted_total)
        print(f"Trajectory {idx}: per-step={reward.per_step} discounted_total={reward.discounted_total:.4f} bonus={reward.done_bonus:.4f}")

    mean_total = sum(totals) / len(totals)
    print(f"\nScored {len(trajectories)} trajectories. Mean discounted total: {mean_total:.4f}")


if __name__ == "__main__":
    main()
