"""
GRPO trainer scaffolding for structured intent disambiguation.

This package provides:
- Shared types/constants for the JSON schema.
- Reward components (R1–R5) as composable functions.
- Replay cache utilities.
- Rollout scheduling helpers.
- A trainer stub showing how to wire everything into Unsloth GRPO.

The code is intentionally lightweight and self-contained so you can swap
implementations or plug into your existing training loop.
"""

__all__ = [
    "schema",
    "reward_components",
    "replay_cache",
    "schedule",
    "trainer_scaffold",
]
