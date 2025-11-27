"""
Lightweight mock PPA KG loader.

The canonical KG spec lives in examples/mock_kg.json. This module loads it and
exposes helpers for scripts/tests.
"""

from pathlib import Path
import json
from typing import Dict, List

MOCK_KG_PATH = Path(__file__).with_suffix(".json")


def load_mock_intents() -> List[dict]:
    with MOCK_KG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


MOCK_INTENTS: List[dict] = load_mock_intents()


def intent_index_by_id() -> Dict[str, dict]:
    return {intent["intent_id"]: intent for intent in MOCK_INTENTS}
