"""
Minimal TRL GRPO integration example using the reward components directly.

Dataset expectation: each item should include at least these columns so the reward
wrappers can pick them up via **kwargs:
  - "gold_intents": list[str]
  - "gold_decision": str
  - "ambiguous_slots": list[str]
  - "reasoning_terms": list[str]
  - optional "known_paths": list[str] for KG whitelist

See https://huggingface.co/docs/trl/main/en/grpo_trainer for the up-to-date API.
"""

from typing import List

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_intent.dataset import load_decision_jsonl
from grpo_intent.reward_components import (
    reward_decision_type,
    reward_intent_similarity,
    reward_json_and_path,
    reward_question_coverage,
    reward_reasoning_quality,
)
from grpo_intent.intent_reward import DecisionRewardConfig, reward_decision_object


def build_trl_grpo_trainer(
    model_name: str,
    train_dataset,
    initial_k: int = 8,
    max_completion_length: int = 64,
    lr: float = 5e-6,
    weight_decay: float = 0.01,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    report_to: str | None = "wandb",
    reward_variant: str = "legacy",
    decision_reward_config: DecisionRewardConfig = DecisionRewardConfig(),
) -> GRPOTrainer:
    """
    Construct a TRL GRPOTrainer with our reward functions attached.

    GRPOTrainer passes batch columns to reward functions via **kwargs. The
    wrappers below pull labels from kwargs so we don't guess trainer internals.

    reward_variant:
      - "legacy": uses the R1–R5 functions based on `final_intents` schema.
      - "decision_object": uses the DecisionObject schema (decision_type,
        picked_intents with path/intent_id, clarification_questions, reasoning_summary).
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

    # Wrappers that look for expected label columns in kwargs.
    def r1(prompts, completions, **kwargs):
        return reward_json_and_path(
            prompts,
            completions,
            gold_intents=kwargs.get("gold_intents"),
            known_paths=kwargs.get("known_paths"),
        )

    def r2(prompts, completions, **kwargs):
        return reward_decision_type(
            prompts,
            completions,
            gold_decisions=kwargs.get("gold_decision", []),
        )

    def r3(prompts, completions, **kwargs):
        return reward_intent_similarity(
            prompts,
            completions,
            gold_intents=kwargs.get("gold_intents", []),
        )

    def r4(prompts, completions, **kwargs):
        return reward_question_coverage(
            prompts,
            completions,
            ambiguous_slots=kwargs.get("ambiguous_slots", []),
        )

    def r5(prompts, completions, **kwargs):
        return reward_reasoning_quality(
            prompts,
            completions,
            key_terms=kwargs.get("reasoning_terms", []),
        )

    def decision_reward(prompts, completions, **kwargs):
        return reward_decision_object(
            prompts=prompts,
            completion_groups=completions,
            gold_intents=kwargs.get("gold_intents"),
            gold_decision=kwargs.get("gold_decision"),
            known_paths=kwargs.get("known_paths"),
            ambiguous_slots=kwargs.get("ambiguous_slots"),
            reasoning_terms=kwargs.get("reasoning_terms"),
            config=decision_reward_config,
        )

    if reward_variant == "decision_object":
        reward_funcs: List = [decision_reward]
    elif reward_variant == "legacy":
        reward_funcs = [r1, r2, r3, r4, r5]
    else:
        raise ValueError(f"Unknown reward_variant: {reward_variant}")

    # ref_model is optional for GRPO; set to None if you don't want KL to a reference.
    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        reward_funcs=reward_funcs,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    return trainer


# Usage sketch (replace train_dataset with your HF dataset with the expected columns):
# trainer = build_trl_grpo_trainer("gpt2", train_dataset)
# trainer.train()


def load_decision_dataset(jsonl_path: str):
    """
    Convenience wrapper to load a DecisionObject JSONL via grpo_intent.dataset.
    """
    return load_decision_jsonl(jsonl_path)
