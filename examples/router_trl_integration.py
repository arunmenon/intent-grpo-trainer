"""
TRL GRPO integration for the router policy.

Dataset expectation (JSONL → HF Dataset):
- "prompt": str (user + context)
- "gold_actions": list[dict] of RouterAction-like entries (oracle trace)
- Optional: "tool_menu": list[dict] matching RouterTool fields
- Optional: "budget", "environment"

Completions are expected to be JSON RouterAction objects. You can override
parse_action_fn to support alternative formats.
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, List

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_intent.router_reward import RouterRewardSettings, RouterSimulator
from grpo_intent.router_schema import RouterAction, RouterActionType, RouterObservation, RouterTool


ParserFn = Callable[[str], List[RouterAction]]


def default_parse_actions(raw_text: str) -> List[RouterAction]:
    """
    Accept either a single RouterAction JSON object or {"actions": [...]} wrapping multiple.
    """
    payload = json.loads(raw_text)
    if isinstance(payload, dict) and "actions" in payload and isinstance(payload["actions"], list):
        items = payload["actions"]
    else:
        items = [payload]

    actions: List[RouterAction] = []
    for item in items:
        actions.append(
            RouterAction(
                decision_id=item.get("decision_id", ""),
                step_index=item.get("step_index", len(actions)),
                action_type=RouterActionType(item.get("action_type", "ANSWER")),
                tool_id=item.get("tool_id"),
                plan_id=item.get("plan_id"),
                args=item.get("args", {}),
                answer=item.get("answer"),
                finish_reason=item.get("finish_reason", "continue"),
                router_confidence=item.get("router_confidence"),
                metadata=item.get("metadata", {}),
            )
        )
    return actions


def build_router_observation(example: dict) -> RouterObservation:
    tool_menu = [RouterTool(**tool) for tool in example.get("tool_menu", [])]
    return RouterObservation(
        request_id=str(example.get("request_id", "")),
        turn_id=int(example.get("turn_id", 0)),
        query=example.get("prompt", ""),
        conversation_history=example.get("conversation_history", []),
        intent_prior=example.get("intent_prior", {}),
        kg_context=example.get("kg_context", {}),
        tool_menu=tool_menu,
        partial_trace=example.get("partial_trace", []),
        budget=example.get("budget", {}),
        environment=example.get("environment", {}),
        metadata=example.get("metadata", {}),
    )


def build_router_grpo_trainer(
    model_name: str,
    train_dataset,
    parse_action_fn: ParserFn = default_parse_actions,
    reward_settings: RouterRewardSettings = RouterRewardSettings(),
    initial_k: int = 4,
    max_completion_length: int = 128,
    lr: float = 5e-6,
    weight_decay: float = 0.01,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    report_to: str | None = "wandb",
) -> GRPOTrainer:
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

    simulator = RouterSimulator(settings=reward_settings)

    def router_reward(prompts: Iterable[str], completion_groups: List[List[str]], **kwargs) -> List[float]:
        rewards: List[float] = []
        dataset_examples = kwargs.get("examples", [])
        for idx, completions in enumerate(completion_groups):
            example = dataset_examples[idx] if idx < len(dataset_examples) else {}
            obs = build_router_observation(example)
            for completion in completions:
                try:
                    actions = parse_action_fn(completion)
                    metrics = simulator.run_episode(obs, actions)
                    rewards.append(metrics.reward)
                except Exception:
                    rewards.append(0.0)
        return rewards

    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        reward_funcs=[router_reward],
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    return trainer
