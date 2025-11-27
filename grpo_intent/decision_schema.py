"""
Parsing helpers for the DecisionObject JSON produced by the model.

Expected completion format:
{
  "decision_type": "pick" | "clarify" | "multi_intent",
  "picked_intents": [
    {"intent_id": "...", "path": ["PPA", "Refunds"], "confidence": 0.9}
  ],
  "multi_intent_group": [
    [{"intent_id": "...", "path": ["PPA", "X"]}],
    [{"intent_id": "...", "path": ["PPA", "Y"]}]
  ],
  "clarification_questions": ["..."],
  "reasoning_summary": "..."
}
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set
import json


ALLOWED_DECISION_TYPES = {"pick", "clarify", "multi_intent"}
REQUIRED_FIELDS = (
    "decision_type",
    "picked_intents",
    "multi_intent_group",
    "clarification_questions",
    "reasoning_summary",
)


@dataclass
class PickedIntent:
    intent_id: Optional[str]
    path: List[str] = field(default_factory=list)
    confidence: Optional[float] = None


@dataclass
class ParsedDecision:
    decision_type: Optional[str]
    picked_intents: List[PickedIntent]
    multi_intent_group: List[List[PickedIntent]]
    clarification_questions: List[str]
    reasoning_summary: str
    raw_text: str
    is_json: bool
    missing_fields: Set[str]
    extra_fields: Set[str]


def _normalize_path(value: Any) -> List[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(v) for v in value]
    if isinstance(value, str):
        # Accept dotted paths as a fallback.
        return [part for part in value.split(".") if part]
    return []


def _parse_picked_intents(raw_items: Any) -> List[PickedIntent]:
    parsed: List[PickedIntent] = []
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return parsed
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        intent_id = item.get("intent_id")
        path = _normalize_path(item.get("path", []))
        confidence = item.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        parsed.append(PickedIntent(intent_id=intent_id, path=path, confidence=confidence))
    return parsed


def try_parse_decision(raw_text: str) -> ParsedDecision:
    """
    Parse a completion into ParsedDecision with lenient fallback to keep rewards robust.
    """
    missing: Set[str] = set()
    extra: Set[str] = set()
    decision_type: Optional[str] = None
    picked_intents: List[PickedIntent] = []
    multi_intent_group: List[List[PickedIntent]] = []
    clarification_questions: List[str] = []
    reasoning_summary = ""
    is_json = True

    try:
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise ValueError("Root must be an object")
    except Exception:
        payload = {}
        is_json = False

    if isinstance(payload, dict):
        extra = set(payload.keys()) - set(REQUIRED_FIELDS)
        for field in REQUIRED_FIELDS:
            if field not in payload:
                missing.add(field)

    if isinstance(payload, dict):
        decision_type_val = payload.get("decision_type")
        if isinstance(decision_type_val, str) and decision_type_val in ALLOWED_DECISION_TYPES:
            decision_type = decision_type_val

        picked_intents = _parse_picked_intents(payload.get("picked_intents", []))

        raw_groups = payload.get("multi_intent_group", [])
        if isinstance(raw_groups, Sequence) and not isinstance(raw_groups, (str, bytes)):
            for group in raw_groups:
                group_parsed = _parse_picked_intents(group)
                if group_parsed:
                    multi_intent_group.append(group_parsed)

        raw_questions = payload.get("clarification_questions", [])
        if isinstance(raw_questions, Sequence) and not isinstance(raw_questions, (str, bytes)):
            clarification_questions = [str(q) for q in raw_questions]

        reasoning_val = payload.get("reasoning_summary", "")
        reasoning_summary = str(reasoning_val) if reasoning_val is not None else ""

    return ParsedDecision(
        decision_type=decision_type,
        picked_intents=picked_intents,
        multi_intent_group=multi_intent_group,
        clarification_questions=clarification_questions,
        reasoning_summary=reasoning_summary,
        raw_text=raw_text,
        is_json=is_json,
        missing_fields=missing,
        extra_fields=extra,
    )


def flatten_intents(parsed: ParsedDecision) -> List[PickedIntent]:
    """
    Return a flat list of all picked intents across single and multi groups.
    """
    combined: List[PickedIntent] = []
    combined.extend(parsed.picked_intents)
    for group in parsed.multi_intent_group:
        combined.extend(group)
    return combined
