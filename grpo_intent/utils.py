"""
Small shared utilities to avoid repeating helper functions across reward modules.
"""

from typing import Any


def extract_text(completion: Any) -> str:
    """
    Normalize completion objects to plain text. Supports string or dict with
    'content'/'text' keys (common in TRL/Unsloth outputs).
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        if "text" in completion:
            return str(completion["text"])
    return str(completion)


def clip_value(value: float, low: float, high: float) -> float:
    """Clamp value into [low, high]."""
    return max(low, min(high, value))
