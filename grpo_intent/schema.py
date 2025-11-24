from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json

# Allowed values for structured outputs.
ALLOWED_DECISION_TYPES: Set[str] = {"pick", "clarify", "multi_intent"}
REQUIRED_FIELDS: Tuple[str, ...] = (
    "decision_type",
    "final_intents",
    "clarifying_questions",
    "reasoning",
)


@dataclass
class ParsedOutput:
    decision_type: Optional[str]
    final_intents: List[str]
    clarifying_questions: List[str]
    reasoning: str
    raw_text: str
    is_json: bool
    missing_fields: Set[str]
    extra_fields: Set[str]


class SchemaViolation(Exception):
    """Raised when a completion cannot be coerced into the schema."""


def try_parse_completion(raw_text: str) -> ParsedOutput:
    """
    Best-effort parse of a completion into the expected JSON schema.

    Returns ParsedOutput with flags indicating validity. This avoids hard
    failures inside reward functions; they can decide how to score partially
    valid outputs.
    """
    missing: Set[str] = set()
    extra: Set[str] = set()
    decision_type: Optional[str] = None
    final_intents: List[str] = []
    clarifying_questions: List[str] = []
    reasoning = ""
    is_json = True

    try:
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise ValueError("Root must be an object")
    except Exception:
        # Fall back to an empty payload for downstream scoring.
        payload = {}
        is_json = False

    if isinstance(payload, dict):
        extra = set(payload.keys()) - set(REQUIRED_FIELDS)
        for field in REQUIRED_FIELDS:
            if field not in payload:
                missing.add(field)

    # Extract fields with lenient type checks.
    decision_type = payload.get("decision_type") if isinstance(payload, dict) else None
    if decision_type not in ALLOWED_DECISION_TYPES:
        decision_type = None

    if isinstance(payload, dict):
        raw_intents = payload.get("final_intents", [])
        if isinstance(raw_intents, Iterable) and not isinstance(raw_intents, (str, bytes)):
            final_intents = [str(x) for x in raw_intents]
        raw_questions = payload.get("clarifying_questions", [])
        if isinstance(raw_questions, Iterable) and not isinstance(raw_questions, (str, bytes)):
            clarifying_questions = [str(x) for x in raw_questions]
        reasoning_val = payload.get("reasoning", "")
        reasoning = str(reasoning_val) if reasoning_val is not None else ""

    return ParsedOutput(
        decision_type=decision_type,
        final_intents=final_intents,
        clarifying_questions=clarifying_questions,
        reasoning=reasoning,
        raw_text=raw_text,
        is_json=is_json,
        missing_fields=missing,
        extra_fields=extra,
    )


def validate_intents_exist(
    intents: Sequence[str],
    known_paths: Optional[Set[str]] = None,
    exists_fn: Optional[Any] = None,
) -> List[bool]:
    """
    Validate whether each intent/path is present in the KG.

    - known_paths: optional whitelist of valid path strings/IDs for quick lookup.
    - exists_fn: optional callable that takes an intent/path and returns bool.
    """
    results: List[bool] = []
    for intent in intents:
        if known_paths is not None:
            results.append(intent in known_paths)
        elif exists_fn is not None:
            results.append(bool(exists_fn(intent)))
        else:
            # If no knowledge provided, treat as unknown (neither penalize nor reward).
            results.append(True)
    return results
