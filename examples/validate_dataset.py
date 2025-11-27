"""
Validate a DecisionObject JSONL dataset against the mock KG.

Checks:
- required columns exist
- gold_intents and known_paths exist in KG
- gold_decision in allowed set
- ambiguous_slots align with required/optional slots of referenced intents (warn only)

Usage:
  python3 examples/validate_dataset.py --data examples/mock_rl_dataset.jsonl --kg examples/mock_kg.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

ALLOWED_DECISIONS = {"pick", "clarify", "multi_intent"}
REQUIRED_FIELDS = [
    "prompt",
    "gold_intents",
    "known_paths",
    "gold_decision",
    "ambiguous_slots",
    "clarification_targets",
    "reasoning_terms",
    "decision_category",
    "difficulty",
]


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_kg(path: Path) -> Dict[str, dict]:
    intents = json.loads(path.read_text(encoding="utf-8"))
    return {i["intent_id"]: i for i in intents}


def normalize_path(path_item) -> List[str]:
    if isinstance(path_item, list):
        return [str(p) for p in path_item]
    if isinstance(path_item, str):
        return [p for p in path_item.split(".") if p]
    return [str(path_item)]


def validate_row(row: dict, kg_index: Dict[str, dict]) -> List[str]:
    errors: List[str] = []
    for key in REQUIRED_FIELDS:
        if key not in row:
            errors.append(f"missing field {key}")
    # intent checks
    gold_intents = row.get("gold_intents", [])
    if not gold_intents:
        errors.append("gold_intents empty")
    for intent_id in gold_intents:
        if intent_id not in kg_index:
            errors.append(f"unknown intent_id {intent_id}")
    # path checks
    known_paths = row.get("known_paths", [])
    for p in known_paths:
        norm = tuple(normalize_path(p))
        if not any(tuple(kg_index[i]["path"]) == norm for i in gold_intents if i in kg_index):
            errors.append(f"known_path {norm} not in KG for gold_intents")
    # decision
    decision = row.get("gold_decision")
    if decision not in ALLOWED_DECISIONS:
        errors.append(f"invalid gold_decision {decision}")
    # ambiguous slot hints
    all_slots: Set[str] = set()
    for intent_id in gold_intents:
        intent = kg_index.get(intent_id)
        if intent:
            all_slots.update(intent.get("required_slots", []))
            all_slots.update(intent.get("optional_slots", []))
    for slot in row.get("ambiguous_slots", []):
        if all_slots and slot not in all_slots:
            errors.append(f"ambiguous slot {slot} not in intent slots")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to JSONL dataset.")
    parser.add_argument("--kg", type=Path, default=Path("examples/mock_kg.json"), help="Path to KG JSON.")
    args = parser.parse_args()

    kg_index = load_kg(args.kg)
    rows = load_jsonl(args.data)

    total = len(rows)
    bad_rows = 0
    for idx, row in enumerate(rows):
        errs = validate_row(row, kg_index)
        if errs:
            bad_rows += 1
            print(f"[{idx}] {row.get('example_id','<no id>')}: {errs}")

    print(f"Checked {total} rows; errors in {bad_rows}")


if __name__ == "__main__":
    main()
