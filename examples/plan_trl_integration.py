"""
TRL GRPO integration for plan-aware (multi-turn) rewards.

Dataset expectation:
- Each row should include:
  - "prompt": str
  - "gold_plan_steps": list[dict] with at least {"node_id": str, "gold_slots": {...}}
  - optional "plan_completed": bool
  - optional "plan_metadata": dict
  - any other fields you want to surface to the completion parser
- Completions are expected to be JSON strings with a "steps" list; see
  `default_parse_completion` for the expected shape. You can override the parser
  to match your model format.

The reward function parses each completion into a PlanTrajectory, combines with
the gold labels from the dataset, and scores via PlanRewardSettings.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Iterable, List, Sequence

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_intent.plan_reward import (
    PlanRewardResult,
    PlanRewardSettings,
    PlanStep,
    PlanTrajectory,
    compute_trajectory_reward,
)
from grpo_intent.plan_schema import PlanSpec


PlanParser = Callable[[str, Sequence[dict], PlanSpec], PlanTrajectory]


def default_parse_completion(raw_text: str, gold_steps: Sequence[dict], plan: PlanSpec) -> PlanTrajectory:
    """
    Lenient parser that expects completion JSON like:
    {
      "steps": [
        {
          "node_id": "decide_refund",
          "pred_slots": {"order_id": "..."},
          "branch_taken": "full",
          "tool_call": {...},
          "state_before": {...},
          "state_after": {...},
          "latency_ms": 123,
          "policy_ok": true
        },
        ...
      ],
      "plan_completed": true
    }
    Gold slots are injected from the dataset's gold_steps.
    """
    try:
        payload = json.loads(raw_text)
    except Exception:
        payload = {}

    steps_data = payload.get("steps", []) if isinstance(payload, dict) else []
    parsed_steps: List[PlanStep] = []
    for idx, gold in enumerate(gold_steps):
        step_payload = steps_data[idx] if idx < len(steps_data) else {}
        parsed_steps.append(
            PlanStep(
                node_id=step_payload.get("node_id", gold.get("node_id")),
                pred_slots=step_payload.get("pred_slots", step_payload.get("slots", {})),
                gold_slots=gold.get("gold_slots", {}),
                branch_taken=step_payload.get("branch_taken"),
                tool_call=step_payload.get("tool_call"),
                state_before=step_payload.get("state_before", {}),
                state_after=step_payload.get("state_after", {}),
                latency_ms=step_payload.get("latency_ms", 0.0),
                policy_ok=bool(step_payload.get("policy_ok", True)),
                t=step_payload.get("t"),
            )
        )

    plan_completed = bool(payload.get("plan_completed", False))
    return PlanTrajectory(steps=parsed_steps, plan_completed=plan_completed)


def build_plan_trl_grpo_trainer(
    model_name: str,
    train_dataset,
    plan_spec: PlanSpec,
    parse_completion_fn: PlanParser = default_parse_completion,
    plan_reward_settings: PlanRewardSettings = PlanRewardSettings(),
    initial_k: int = 4,
    max_completion_length: int = 128,
    lr: float = 5e-6,
    weight_decay: float = 0.01,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    report_to: str | None = "wandb",
) -> GRPOTrainer:
    """
    Construct a TRL GRPOTrainer wired to plan rewards.

    The reward_fn parses completions into PlanTrajectory objects, pairs them
    with gold steps from the dataset (kwargs["gold_plan_steps"]), and returns
    discounted totals as rewards. For finer credit assignment, you can modify
    this to emit per-step advantages and align to token logprobs.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    grpo_config = GRPOConfig(
        num_generations=initial_k,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=weight_decay,
        report_to=report_to,
    )

    def plan_reward(prompts: Iterable[str], completion_groups: List[List[str]], **kwargs) -> List[float]:
        rewards: List[float] = []
        gold_steps_batches: List[Sequence[dict]] = kwargs.get("gold_plan_steps", [])
        plan_completed_flags: List[bool] = kwargs.get("plan_completed", [])

        for prompt_idx, completions in enumerate(completion_groups):
            gold_steps = gold_steps_batches[prompt_idx] if prompt_idx < len(gold_steps_batches) else []
            plan_completed = (
                bool(plan_completed_flags[prompt_idx]) if prompt_idx < len(plan_completed_flags) else False
            )
            for completion in completions:
                try:
                    traj = parse_completion_fn(completion, gold_steps, plan_spec)
                    # If dataset provided completion flag, override parsed value to keep labels authoritative.
                    if plan_completed:
                        traj.plan_completed = True
                    result: PlanRewardResult = compute_trajectory_reward(traj, plan_spec, plan_reward_settings)
                    rewards.append(result.discounted_total)
                except Exception:
                    rewards.append(0.0)
        return rewards

    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        reward_funcs=[plan_reward],
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    return trainer
