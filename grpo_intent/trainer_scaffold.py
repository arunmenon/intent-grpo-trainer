"""
Illustrative wiring of rewards, replay cache, and rollout scheduling with Unsloth GRPO.

This is a scaffold, not an executable trainer. Swap the placeholders with your
actual Unsloth GRPOTrainer, generate_fn, and dataset loader.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from .replay_cache import ExperienceCache
from .reward_components import (
    RewardSettings,
    reward_decision_type,
    reward_intent_similarity,
    reward_json_and_path,
    reward_question_coverage,
    reward_reasoning_quality,
    sum_with_weights,
)
from .schedule import RolloutSchedule


# Type aliases for clarity.
CompletionGroup = List[str]
RewardFunc = Callable[..., List[float]]


@dataclass
class TrainerHooks:
    """
    Replace these hooks with real functions when integrating:
    - generate_fn: callable(prompt_text, num, max_len, temperature) -> List[str]
    - backward_update_fn: callable(batch_prompts, completion_groups, advantages) -> None
    """

    generate_fn: Callable[[str, int, int, float], CompletionGroup]
    backward_update_fn: Callable[[Sequence[Any], List[CompletionGroup], List[float]], None]


@dataclass
class IntentTrainerConfig:
    reward_settings: RewardSettings = RewardSettings()
    reward_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0.5, 0.2])
    topk_per_prompt: int = 3
    cache_mix_ratio: float = 0.5
    schedule: RolloutSchedule = RolloutSchedule(total_steps=1000)


class IntentGRPOTrainer:
    """
    High-level orchestrator to:
    - mix cached + fresh generations,
    - compute rewards (R1–R5),
    - sum with weights and hand advantages to your GRPO trainer.
    """

    def __init__(self, config: IntentTrainerConfig, hooks: TrainerHooks):
        self.config = config
        self.hooks = hooks
        self.cache = ExperienceCache(topk_per_prompt=config.topk_per_prompt)

    def _generate_with_cache(
        self, prompt_id: Any, prompt_text: str, k: int, max_len: int, temperature: float
    ) -> CompletionGroup:
        cached, fresh_needed = self.cache.sample_mix(prompt_id, total=k, cache_mix_ratio=self.config.cache_mix_ratio)
        fresh = self.hooks.generate_fn(prompt_text, fresh_needed, max_len, temperature)
        return cached + fresh

    def train_step(
        self,
        step: int,
        prompts: Sequence[Any],
        prompt_texts: Sequence[str],
        gold_intents: Sequence[Sequence[str]],
        gold_decisions: Sequence[str],
        ambiguous_slots: Sequence[Sequence[str]],
        reasoning_terms: Sequence[Sequence[str]],
        path_whitelists: Sequence[Sequence[str]] | None = None,
    ) -> Dict[str, Any]:
        """
        Single training step over a batch of prompts.
        """
        k, max_len, temperature = self.config.schedule.at(step)

        # 1) Generate completions (cached + fresh).
        completion_groups: List[CompletionGroup] = []
        for pid, text in zip(prompts, prompt_texts):
            group = self._generate_with_cache(pid, text, k=k, max_len=max_len, temperature=temperature)
            completion_groups.append(group)

        # 2) Compute component rewards.
        r1 = reward_json_and_path(
            prompts, completion_groups, gold_intents=gold_intents, known_paths=path_whitelists, settings=self.config.reward_settings
        )
        r2 = reward_decision_type(prompts, completion_groups, gold_decisions=gold_decisions, settings=self.config.reward_settings)
        r3 = reward_intent_similarity(prompts, completion_groups, gold_intents=gold_intents, settings=self.config.reward_settings)
        r4 = reward_question_coverage(prompts, completion_groups, ambiguous_slots=ambiguous_slots, settings=self.config.reward_settings)
        r5 = reward_reasoning_quality(prompts, completion_groups, key_terms=reasoning_terms, settings=self.config.reward_settings)

        component_list = [r1, r2, r3, r4, r5]
        total_rewards = sum_with_weights(component_list, self.config.reward_weights, self.config.reward_settings)

        # 3) Compute advantages via group-wise normalization.
        advantages: List[float] = []
        cursor = 0
        for group in completion_groups:
            group_rewards = total_rewards[cursor : cursor + len(group)]
            cursor += len(group)
            if not group_rewards:
                continue
            mean = sum(group_rewards) / len(group_rewards)
            # Avoid zero std; add small epsilon.
            var = sum((r - mean) ** 2 for r in group_rewards) / len(group_rewards)
            std = (var ** 0.5) or 1e-6
            advantages.extend([(r - mean) / std for r in group_rewards])

        # 4) Backward/update hook (delegates to actual GRPO trainer).
        self.hooks.backward_update_fn(prompts, completion_groups, advantages)

        # 5) Update cache with rewards to reinforce good samples.
        cursor = 0
        for pid, group in zip(prompts, completion_groups):
            group_rewards = total_rewards[cursor : cursor + len(group)]
            cursor += len(group)
            self.cache.update(pid, group, group_rewards)

        return {
            "k": k,
            "max_len": max_len,
            "temperature": temperature,
            "mean_reward": sum(total_rewards) / max(len(total_rewards), 1),
        }
