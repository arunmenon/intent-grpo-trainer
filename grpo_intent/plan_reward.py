"""
Composable micro- and trajectory-level rewards for multi-turn plan execution.

This sits alongside the single-turn reward modules and reuses similar patterns:
- pure helper functions for each micro-signal,
- dataclasses for config and runtime records,
- clipping to keep rewards bounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .plan_schema import PlanBranch, PlanSpec
from .utils import clip_value


@dataclass
class PlanStep:
    """
    Per-turn record describing what the agent did and the surrounding state.

    - pred_slots / gold_slots: string-keyed slot dicts for F1.
    - branch_taken: branch_id chosen by the agent (if applicable).
    - tool_call: raw tool call payload; validated by the node's ToolSpec validator.
    - state_before/state_after: environment snapshots for pre/post checks.
    - latency_ms: wall-clock latency for this step.
    - policy_ok: True if no policy/PII violation was observed.
    - t: optional step index used for discounting; falls back to list index.
    """

    node_id: str
    pred_slots: Mapping[str, Any]
    gold_slots: Mapping[str, Any]
    branch_taken: Optional[str] = None
    tool_call: Any = None
    state_before: Mapping[str, Any] = field(default_factory=dict)
    state_after: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    policy_ok: bool = True
    t: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanTrajectory:
    """Sequence of PlanStep records produced from one prompt/plan."""

    steps: List[PlanStep]
    plan_completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanRewardSettings:
    """
    Tunable weights and shaping constants for plan rewards.

    - component_clip/total_clip bound individual step scores and the final return.
    - required_slot_weight/optional_slot_weight let you favor must-have slots.
    - discount_done controls whether the completion bonus is discounted by gamma.
    """

    w_slot: float = 0.35
    w_prec: float = 0.1
    w_tool: float = 0.15
    w_branch: float = 0.1
    w_policy: float = 0.1
    w_latency: float = 0.05
    w_effect: float = 0.1
    w_done: float = 1.0
    gamma: float = 0.97
    tau_ms: float = 1500.0
    required_slot_weight: float = 1.0
    optional_slot_weight: float = 0.5
    component_clip: float = 5.0
    total_clip: float = 10.0
    discount_done: bool = True


@dataclass
class PlanRewardResult:
    """Aggregated reward outputs for a rollout."""

    per_step: List[float]
    discounted_total: float
    done_bonus: float
    raw_total: float


def slot_f1(
    pred_slots: Mapping[str, Any],
    gold_slots: Mapping[str, Any],
    required_slots: Sequence[str],
    optional_slots: Sequence[str],
    required_weight: float,
    optional_weight: float,
) -> float:
    """
    Weighted F1 to give required slots more importance than optional ones.
    """
    required = set(required_slots)
    optional = set(optional_slots)
    keys = set(pred_slots) | set(gold_slots)

    tp = fp = fn = 0.0
    for key in keys:
        weight = required_weight if key in required else optional_weight
        gold_val = gold_slots.get(key)
        pred_val = pred_slots.get(key)
        if pred_val is not None and gold_val is not None and pred_val == gold_val:
            tp += weight
        elif pred_val is not None and (gold_val is None or pred_val != gold_val):
            fp += weight
        elif pred_val is None and gold_val is not None:
            fn += weight

    denom_prec = tp + fp + 1e-9
    denom_rec = tp + fn + 1e-9
    prec = tp / denom_prec
    rec = tp / denom_rec
    return (2 * prec * rec) / (prec + rec + 1e-9)


def preconditions_ok(state_before: Mapping[str, Any], preconditions: Sequence) -> float:
    if not preconditions:
        return 1.0
    try:
        return 1.0 if all(check(state_before) for check in preconditions) else 0.0
    except Exception:
        return 0.0


def effects_hold(state_before: Mapping[str, Any], state_after: Mapping[str, Any], effects: Sequence) -> float:
    if not effects:
        return 1.0
    try:
        return 1.0 if all(check(state_before, state_after) for check in effects) else 0.0
    except Exception:
        return 0.0


def tool_valid(tool_call: Any, node_tool) -> float:
    if node_tool is None or node_tool.validator is None:
        # Neutral if no validator is defined.
        return 1.0
    try:
        return float(node_tool.validator(tool_call))
    except Exception:
        return 0.0


def branch_correct(branch_taken: Optional[str], branches: Sequence[PlanBranch], state_before: Mapping[str, Any]) -> float:
    if not branches:
        return 1.0
    try:
        truthy = [branch.branch_id for branch in branches if branch.guard(state_before)]
    except Exception:
        truthy = []
    if not truthy:
        return 0.0
    return 1.0 if branch_taken in truthy else 0.0


def latency_reward(latency_ms: float, tau_ms: float) -> float:
    latency_ms = max(latency_ms or 0.0, 0.0)
    tau_ms = max(tau_ms, 1e-6)
    return math.exp(-latency_ms / tau_ms)


def compute_step_reward(step: PlanStep, plan: PlanSpec, cfg: PlanRewardSettings) -> float:
    node = plan.node(step.node_id)
    score = 0.0
    score += cfg.w_slot * slot_f1(
        step.pred_slots,
        step.gold_slots,
        required_slots=node.required_slots,
        optional_slots=node.optional_slots,
        required_weight=cfg.required_slot_weight,
        optional_weight=cfg.optional_slot_weight,
    )
    score += cfg.w_prec * preconditions_ok(step.state_before, node.preconditions)
    score += cfg.w_tool * tool_valid(step.tool_call, node.tool_spec)
    score += cfg.w_branch * branch_correct(step.branch_taken, node.branches, step.state_before)
    score += cfg.w_policy * (1.0 if step.policy_ok else 0.0)
    score += cfg.w_latency * latency_reward(step.latency_ms, cfg.tau_ms)
    score += cfg.w_effect * effects_hold(step.state_before, step.state_after, node.effects)
    return clip_value(score, -cfg.component_clip, cfg.component_clip)


def compute_trajectory_reward(trajectory: PlanTrajectory, plan: PlanSpec, cfg: PlanRewardSettings) -> PlanRewardResult:
    """
    Aggregate discounted per-step rewards and an optional completion bonus.
    """
    per_step: List[float] = []
    discounted_total = 0.0

    for idx, step in enumerate(trajectory.steps):
        step_reward = compute_step_reward(step, plan, cfg)
        per_step.append(step_reward)
        step_index = step.t if step.t is not None else idx
        discounted_total += (cfg.gamma ** step_index) * step_reward

    done_bonus = 0.0
    if trajectory.plan_completed:
        bonus_discount = cfg.gamma ** len(trajectory.steps) if cfg.discount_done else 1.0
        done_bonus = cfg.w_done * bonus_discount
        discounted_total += done_bonus

    total_with_bonus = discounted_total
    return PlanRewardResult(
        per_step=per_step,
        discounted_total=clip_value(total_with_bonus, -cfg.total_clip, cfg.total_clip),
        done_bonus=done_bonus,
        raw_total=discounted_total - done_bonus,
    )
