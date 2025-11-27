"""
Lightweight synthetic PPA KG loader.

The canonical KG spec lives in examples/synthetic_kg.json. This module loads it and
exposes helpers for scripts/tests.
"""

from pathlib import Path
import json
from typing import Dict, List

SYNTHETIC_KG_PATH = Path(__file__).with_suffix(".json")


def load_synthetic_intents() -> List[dict]:
    with SYNTHETIC_KG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


SYNTHETIC_INTENTS: List[dict] = load_synthetic_intents()


def intent_index_by_id() -> Dict[str, dict]:
    return {intent["intent_id"]: intent for intent in SYNTHETIC_INTENTS}
