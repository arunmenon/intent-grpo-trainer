"""
Modular reward components (R1–R5) for structured intent disambiguation.

Each reward function takes:
- prompts: a list of prompt texts or metadata (unused except for symmetry).
- completion_groups: list of lists; each inner list contains the completions for one prompt.
- labels/ground truth specific to the reward (e.g., gold intents).

They return a flat list of scores, one per completion, in the same order as
iteration over completion_groups then each completion inside that group.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from .schema import (
    ALLOWED_DECISION_TYPES,
    ParsedOutput,
    try_parse_completion,
    validate_intents_exist,
)


def _extract_text(completion: Any) -> str:
    """Handle Unsloth/TRL completion formats (str or dict with content/text)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        if "text" in completion:
            return str(completion["text"])
    return str(completion)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _flatten(groups: List[List[Any]]) -> List[Any]:
    flat: List[Any] = []
    for group in groups:
        flat.extend(group)
    return flat


@dataclass
class RewardSettings:
    component_clip: float = 5.0
    total_clip: float = 10.0
    # R1
    r1_format_bonus: float = 1.5
    r1_missing_penalty: float = -0.5
    r1_invalid_json_penalty: float = -2.0
    r1_decision_penalty: float = -1.0
    r1_extra_field_penalty: float = -0.1
    r1_invalid_intent_penalty: float = -0.5
    r1_match_weight: float = 2.0
    # R2
    r2_correct: float = 1.0
    r2_incorrect: float = -1.0
    # R3
    r3_jaccard_scale: float = 2.0
    r3_order_penalty: float = -0.5
    r3_extra_intent_penalty: float = -0.5
    # R4
    r4_slot_reward: float = 1.0
    r4_irrelevant_penalty: float = -1.0
    # R5
    r5_keyword_reward: float = 0.2
    r5_miss_penalty: float = -0.2


def reward_json_and_path(
    prompts: Sequence[Any],
    completion_groups: List[List[Any]],
    gold_intents: Optional[List[Sequence[str]]] = None,
    known_paths: Optional[Sequence[Optional[Sequence[str]]]] = None,
    exists_fn: Optional[Callable[[str], bool]] = None,
    settings: RewardSettings = RewardSettings(),
) -> List[float]:
    """
    R1: JSON validity + KG/path correctness + intent overlap (partial credit).
    """
    scores: List[float] = []

    for idx, group in enumerate(completion_groups):
        gold_set = set(gold_intents[idx]) if gold_intents and idx < len(gold_intents) else set()
        path_whitelist = (
            set(known_paths[idx]) if known_paths and idx < len(known_paths) and known_paths[idx] else None
        )

        for completion in group:
            text = _extract_text(completion)
            parsed: ParsedOutput = try_parse_completion(text)
            score = 0.0

            if not parsed.is_json:
                score += settings.r1_invalid_json_penalty
            else:
                score += settings.r1_format_bonus

            if parsed.missing_fields:
                score += settings.r1_missing_penalty * len(parsed.missing_fields)
            if parsed.extra_fields:
                score += settings.r1_extra_field_penalty * len(parsed.extra_fields)
            if parsed.decision_type is None:
                score += settings.r1_decision_penalty

            validity_flags = validate_intents_exist(parsed.final_intents, known_paths=path_whitelist, exists_fn=exists_fn)
            invalid_count = sum(1 for ok in validity_flags if not ok)
            if invalid_count:
                score += settings.r1_invalid_intent_penalty * invalid_count

            if gold_set:
                pred_set = set(parsed.final_intents)
                intersection = len(pred_set & gold_set)
                union = len(pred_set | gold_set)
                jaccard = intersection / union if union > 0 else 0.0
                score += settings.r1_match_weight * jaccard

            scores.append(_clip(score, -settings.component_clip, settings.component_clip))

    return scores


def reward_decision_type(
    prompts: Sequence[Any],
    completion_groups: List[List[Any]],
    gold_decisions: Sequence[str],
    settings: RewardSettings = RewardSettings(),
) -> List[float]:
    """R2: Correct decision_type vs. ground truth."""
    scores: List[float] = []
    for idx, group in enumerate(completion_groups):
        correct = gold_decisions[idx] if idx < len(gold_decisions) else None
        for completion in group:
            text = _extract_text(completion)
            parsed = try_parse_completion(text)
            if parsed.decision_type is None or correct is None:
                scores.append(settings.r2_incorrect)
                continue
            scores.append(settings.r2_correct if parsed.decision_type == correct else settings.r2_incorrect)
    return scores


def reward_intent_similarity(
    prompts: Sequence[Any],
    completion_groups: List[List[Any]],
    gold_intents: Sequence[Sequence[str]],
    settings: RewardSettings = RewardSettings(),
) -> List[float]:
    """
    R3: Jaccard similarity + ordering penalty for final_intents.
    """
    scores: List[float] = []
    for idx, group in enumerate(completion_groups):
        gold_list = list(gold_intents[idx]) if idx < len(gold_intents) else []
        gold_set = set(gold_list)
        for completion in group:
            text = _extract_text(completion)
            parsed = try_parse_completion(text)
            pred_list = parsed.final_intents
            pred_set = set(pred_list)

            inter = len(pred_set & gold_set)
            union = len(pred_set | gold_set)
            jaccard = inter / union if union > 0 else 0.0
            score = jaccard * settings.r3_jaccard_scale

            # Penalize extra intents when we know gold.
            extras = len([p for p in pred_list if p not in gold_set])
            if extras:
                score += settings.r3_extra_intent_penalty * extras

            # Ordering penalty if lengths match.
            if len(pred_list) == len(gold_list) and gold_list:
                mismatches = sum(
                    1 for i in range(len(gold_list)) if i < len(pred_list) and pred_list[i] != gold_list[i]
                )
                score += settings.r3_order_penalty * mismatches

            scores.append(_clip(score, -settings.component_clip, settings.component_clip))
    return scores


def reward_question_coverage(
    prompts: Sequence[Any],
    completion_groups: List[List[Any]],
    ambiguous_slots: Sequence[Sequence[str]],
    settings: RewardSettings = RewardSettings(),
) -> List[float]:
    """
    R4: Clarifying question coverage for ambiguous slots.

    - ambiguous_slots: list per prompt of slot keywords the question should address.
    """
    scores: List[float] = []
    for idx, group in enumerate(completion_groups):
        slots = [s.lower() for s in ambiguous_slots[idx]] if idx < len(ambiguous_slots) else []
        for completion in group:
            text = _extract_text(completion)
            parsed = try_parse_completion(text)
            if parsed.decision_type != "clarify":
                scores.append(0.0)
                continue

            question_text = " ".join(parsed.clarifying_questions).lower()
            covered = sum(1 for slot in slots if slot in question_text)
            if slots:
                score = covered * settings.r4_slot_reward
                if covered == 0:
                    score += settings.r4_irrelevant_penalty
            else:
                # No known ambiguity; neutral.
                score = 0.0
            scores.append(_clip(score, -settings.component_clip, settings.component_clip))
    return scores


def reward_reasoning_quality(
    prompts: Sequence[Any],
    completion_groups: List[List[Any]],
    key_terms: Sequence[Sequence[str]],
    settings: RewardSettings = RewardSettings(),
) -> List[float]:
    """
    R5: Light-weight reasoning check.

    - key_terms: per prompt keywords that should appear in reasoning (e.g., slots/intents).
    """
    scores: List[float] = []
    for idx, group in enumerate(completion_groups):
        terms = [t.lower() for t in key_terms[idx]] if idx < len(key_terms) else []
        for completion in group:
            text = _extract_text(completion)
            parsed = try_parse_completion(text)
            reasoning = parsed.reasoning.lower()
            hit = sum(1 for term in terms if term in reasoning)
            miss = len(terms) - hit
            score = hit * settings.r5_keyword_reward - miss * settings.r5_miss_penalty if terms else 0.0
            scores.append(_clip(score, -settings.component_clip, settings.component_clip))
    return scores


def sum_with_weights(components: List[List[float]], weights: Sequence[float], settings: RewardSettings) -> List[float]:
    """
    Combine component rewards with weights and final clipping.
    """
    if len(weights) != len(components):
        raise ValueError("weights must align with components")

    flat_lengths = {len(comp) for comp in components}
    if len(flat_lengths) != 1:
        raise ValueError("All component lists must have the same length")

    total: List[float] = [0.0 for _ in range(flat_lengths.pop())]
    for comp, weight in zip(components, weights):
        for i, val in enumerate(comp):
            total[i] += weight * val

    return [_clip(t, -settings.total_clip, settings.total_clip) for t in total]
