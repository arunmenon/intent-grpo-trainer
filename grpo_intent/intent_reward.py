"""
Reward functions for the DecisionObject schema used in PPA intent routing.

Components:
- R1: JSON/schema validity
- R2: KG/path validity
- R3: decision_type correctness
- R4: intent match
- R5: clarification + reasoning coverage
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

from .decision_schema import ALLOWED_DECISION_TYPES, ParsedDecision, PickedIntent, flatten_intents, try_parse_decision
from .utils import clip_value, extract_text


def _path_key(path: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(p) for p in path)


def _normalize_known_paths(known_paths: Optional[Sequence[Any]]) -> Set[Tuple[str, ...]]:
    normalized: Set[Tuple[str, ...]] = set()
    if not known_paths:
        return normalized
    for path in known_paths:
        if isinstance(path, str):
            parts = [p for p in path.split(".") if p]
            normalized.add(tuple(parts) if parts else (path,))
        elif isinstance(path, Iterable) and not isinstance(path, (bytes, str)):
            normalized.add(_path_key(path))
    return normalized


def _intent_ids(intents: Iterable[PickedIntent]) -> List[str]:
    return [i.intent_id for i in intents if i.intent_id]


@dataclass
class DecisionRewardConfig:
    component_clip: float = 5.0
    total_clip: float = 10.0

    json_bonus: float = 1.0
    invalid_json_penalty: float = -2.0
    missing_field_penalty: float = -0.5
    extra_field_penalty: float = -0.2
    invalid_decision_penalty: float = -1.0
    missing_question_penalty: float = -1.0
    missing_multi_group_penalty: float = -1.0

    path_valid_reward: float = 0.5
    path_invalid_penalty: float = -0.5

    decision_correct: float = 1.0
    decision_incorrect: float = -1.0

    intent_match_scale: float = 2.0
    extra_intent_penalty: float = -0.5

    clarify_slot_reward: float = 0.5
    clarify_slot_miss_penalty: float = -0.5
    reasoning_hit_reward: float = 0.2
    reasoning_miss_penalty: float = -0.2

    weights: Tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.25, 0.15)


def _r1_schema(parsed: ParsedDecision, config: DecisionRewardConfig) -> float:
    score = config.json_bonus if parsed.is_json else config.invalid_json_penalty
    score += config.missing_field_penalty * len(parsed.missing_fields)
    score += config.extra_field_penalty * len(parsed.extra_fields)
    if parsed.decision_type not in ALLOWED_DECISION_TYPES:
        score += config.invalid_decision_penalty
    if parsed.decision_type == "clarify" and not parsed.clarification_questions:
        score += config.missing_question_penalty
    if parsed.decision_type == "multi_intent" and not parsed.multi_intent_group:
        score += config.missing_multi_group_penalty
    return clip_value(score, -config.component_clip, config.component_clip)


def _r2_paths(parsed: ParsedDecision, known_paths: Set[Tuple[str, ...]], config: DecisionRewardConfig) -> float:
    intents = flatten_intents(parsed)
    if not intents or not known_paths:
        return 0.0
    valid = 0
    invalid = 0
    for intent in intents:
        path_key = _path_key(intent.path)
        if not path_key:
            invalid += 1
            continue
        if path_key in known_paths:
            valid += 1
        else:
            invalid += 1
    score = valid * config.path_valid_reward + invalid * config.path_invalid_penalty
    return clip_value(score, -config.component_clip, config.component_clip)


def _r3_decision(parsed: ParsedDecision, gold: Optional[str], config: DecisionRewardConfig) -> float:
    if not gold:
        return 0.0
    if parsed.decision_type == gold:
        return config.decision_correct
    return config.decision_incorrect


def _r4_intents(parsed: ParsedDecision, gold_intents: Set[str], config: DecisionRewardConfig) -> float:
    if not gold_intents:
        return 0.0
    pred_ids = set(_intent_ids(flatten_intents(parsed)))
    inter = len(pred_ids & gold_intents)
    union = len(pred_ids | gold_intents)
    jaccard = inter / union if union else 0.0
    extras = len([pid for pid in pred_ids if pid not in gold_intents])
    score = jaccard * config.intent_match_scale
    if extras:
        score += config.extra_intent_penalty * extras
    return clip_value(score, -config.component_clip, config.component_clip)


def _r5_clarify_reasoning(parsed: ParsedDecision, slots: List[str], terms: List[str], config: DecisionRewardConfig) -> float:
    score = 0.0
    if slots:
        question_text = " ".join(parsed.clarification_questions).lower()
        normalized_slots = [s.lower() for s in slots]
        hits = sum(1 for slot in normalized_slots if slot in question_text)
        misses = len(normalized_slots) - hits
        score += hits * config.clarify_slot_reward
        if hits == 0:
            score += config.clarify_slot_miss_penalty
        else:
            score -= misses * abs(config.clarify_slot_miss_penalty) * 0.5
    if terms:
        reasoning = parsed.reasoning_summary.lower()
        normalized_terms = [t.lower() for t in terms]
        hits = sum(1 for term in normalized_terms if term in reasoning)
        misses = len(normalized_terms) - hits
        score += hits * config.reasoning_hit_reward
        score -= misses * abs(config.reasoning_miss_penalty)
    return clip_value(score, -config.component_clip, config.component_clip)


def reward_decision_object(
    prompts: Sequence[Any],
    completion_groups: List[List[Any]],
    gold_intents: Optional[Sequence[Sequence[str]]] = None,
    gold_decision: Optional[Sequence[str]] = None,
    known_paths: Optional[Sequence[Sequence[Any]]] = None,
    ambiguous_slots: Optional[Sequence[Sequence[str]]] = None,
    reasoning_terms: Optional[Sequence[Sequence[str]]] = None,
    config: DecisionRewardConfig = DecisionRewardConfig(),
) -> List[float]:
    """
    Compute rewards for DecisionObject outputs. Mirrors the completion_groups layout.
    """
    rewards: List[float] = []
    w1, w2, w3, w4, w5 = config.weights

    for idx, group in enumerate(completion_groups):
        gold_set = set(gold_intents[idx]) if gold_intents and idx < len(gold_intents) else set()
        gold_decision_val = gold_decision[idx] if gold_decision and idx < len(gold_decision) else None
        known_path_set = _normalize_known_paths(known_paths[idx]) if known_paths and idx < len(known_paths) else set()
        slots = list(ambiguous_slots[idx]) if ambiguous_slots and idx < len(ambiguous_slots) else []
        reason_terms = list(reasoning_terms[idx]) if reasoning_terms and idx < len(reasoning_terms) else []

        for completion in group:
            text = extract_text(completion)
            parsed = try_parse_decision(text)

            r1 = _r1_schema(parsed, config)
            r2 = _r2_paths(parsed, known_path_set, config)
            r3 = _r3_decision(parsed, gold_decision_val, config)
            r4 = _r4_intents(parsed, gold_set, config)
            r5 = _r5_clarify_reasoning(parsed, slots, reason_terms, config)

            total = w1 * r1 + w2 * r2 + w3 * r3 + w4 * r4 + w5 * r5
            rewards.append(clip_value(total, -config.total_clip, config.total_clip))

    return rewards
