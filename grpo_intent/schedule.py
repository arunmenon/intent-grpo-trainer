"""
Rollout scheduling utilities (progressively decay k, increase length, adjust temperature).
"""

from dataclasses import dataclass


def _interp(start: float, end: float, fraction: float) -> float:
    fraction = max(0.0, min(1.0, fraction))
    return start + (end - start) * fraction


@dataclass
class RolloutSchedule:
    total_steps: int
    k_start: int = 8
    k_end: int = 2
    max_len_start: int = 64
    max_len_end: int = 256
    temp_start: float = 1.0
    temp_end: float = 0.75

    def at(self, step: int) -> tuple[int, int, float]:
        """
        Compute (num_generations, max_completion_length, temperature) for the given step.
        Linear interpolation between start and end values.
        """
        fraction = min(max(step / max(self.total_steps, 1), 0.0), 1.0)
        k = round(_interp(self.k_start, self.k_end, fraction))
        max_len = int(_interp(self.max_len_start, self.max_len_end, fraction))
        temp = _interp(self.temp_start, self.temp_end, fraction)
        return max(k, 1), max(max_len, 1), max(temp, 0.1)


def boost_for_hard_prompts(base_k: int, recent_reward: float, threshold: float = 0.3, max_boost: int = 3) -> int:
    """
    Simple heuristic to allocate more generations to underperforming prompts.
    """
    if recent_reward < threshold:
        return min(base_k + max_boost, base_k * 2)
    return base_k
