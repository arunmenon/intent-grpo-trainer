"""
Dataset utilities for DecisionObject-style RL training.

- load_decision_jsonl: load JSONL where each row includes prompt + reward metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping

from datasets import Dataset


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


def _read_jsonl(path: Path) -> List[MutableMapping]:
    records: List[MutableMapping] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def load_decision_jsonl(jsonl_path: str | Path) -> Dataset:
    """
    Load a JSONL file into a Hugging Face Dataset, ensuring required fields exist.
    """
    path = Path(jsonl_path)
    records = _read_jsonl(path)
    for idx, row in enumerate(records):
        for key in REQUIRED_FIELDS:
            if key not in row:
                row[key] = [] if key.endswith("s") else ""
    return Dataset.from_list(records)
