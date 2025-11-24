"""
Simple top-K replay cache per prompt for GRPO rollouts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple
import heapq
import random


@dataclass
class CachedCompletion:
    reward: float
    text: str
    metadata: dict = field(default_factory=dict)


class ExperienceCache:
    """
    Maintains top-K completions per prompt.
    """

    def __init__(self, topk_per_prompt: int = 3):
        self.topk_per_prompt = topk_per_prompt
        self._store: Dict[Any, List[Tuple[float, CachedCompletion]]] = {}

    def get(self, prompt_id: Any, k: int) -> List[CachedCompletion]:
        """Return up to k cached completions (highest reward first)."""
        items = self._store.get(prompt_id, [])
        # Stored as min-heap of (-reward, CachedCompletion) for max ordering.
        sorted_items = sorted(items, key=lambda x: x[0])
        completions = [cc for _, cc in sorted_items[:k]]
        return completions

    def update(self, prompt_id: Any, completions: Iterable[str], rewards: Iterable[float]) -> None:
        """
        Insert completions with their rewards into the cache, keeping only top-K.
        """
        heap = self._store.setdefault(prompt_id, [])
        for text, reward in zip(completions, rewards):
            entry = CachedCompletion(reward=reward, text=text)
            heapq.heappush(heap, (-reward, entry))
            if len(heap) > self.topk_per_prompt:
                heapq.heappop(heap)

    def merge_from(self, other: "ExperienceCache") -> None:
        """
        Merge caches (useful for cross-rank sync). Keeps top-K per prompt.
        """
        for prompt_id, heap in other._store.items():
            for reward_neg, entry in heap:
                self.update(prompt_id, [entry.text], [-reward_neg])

    def sample_mix(self, prompt_id: Any, total: int, cache_mix_ratio: float) -> Tuple[List[str], int]:
        """
        Return cached texts to use and number of fresh completions to generate.
        Ensures at least one fresh sample when total > 0.
        """
        cached_count = int(total * cache_mix_ratio)
        cached_items = self.get(prompt_id, cached_count)
        cached_texts = [c.text for c in cached_items]
        fresh_needed = max(total - len(cached_texts), 1 if total > 0 else 0)
        return cached_texts, fresh_needed
