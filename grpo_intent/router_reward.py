"""
Reward shaping and a minimal simulator for the router policy.

R =  w_success * success
   - w_latency * (latency_ms / 1000)
   - w_cost    * cost_usd
   - w_viol    * violations
   - w_big     * big_model_calls
   - w_steps   * max(0, num_steps - steps_min)

Weights are configurable via RouterRewardSettings.
"""

from dataclasses import dataclass
from typing import List

from .router_schema import RouterAction, RouterActionType, RouterObservation, TraceStep


@dataclass
class RouterRewardSettings:
    w_success: float = 3.0
    w_latency: float = 0.15  # per second
    w_cost: float = 50.0     # per USD
    w_viol: float = 2.0
    w_big: float = 0.5
    w_steps: float = 0.1
    steps_min: int = 2
    big_model_latency_ms: float = 400.0
    big_model_cost_usd: float = 0.002
    clip_low: float = -5.0
    clip_high: float = 5.0


@dataclass
class RouterEpisodeMetrics:
    success: int
    latency_ms: float
    cost_usd: float
    violations: int
    big_model_calls: int
    num_steps: int
    reward: float


def compute_reward(metrics: RouterEpisodeMetrics, settings: RouterRewardSettings) -> float:
    r = 0.0
    r += settings.w_success * metrics.success
    r -= settings.w_latency * (metrics.latency_ms / 1000.0)
    r -= settings.w_cost * metrics.cost_usd
    r -= settings.w_viol * metrics.violations
    r -= settings.w_big * metrics.big_model_calls
    r -= settings.w_steps * max(0, metrics.num_steps - settings.steps_min)
    return max(settings.clip_low, min(settings.clip_high, r))


class RouterSimulator:
    """
    Minimal loop to execute router actions against a tool menu stub for demos/tests.
    """

    def __init__(self, settings: RouterRewardSettings | None = None):
        self.settings = settings or RouterRewardSettings()

    def is_big_model(self, action: RouterAction, obs: RouterObservation) -> bool:
        """Heuristic: big model if latency/cost exceed thresholds."""
        tool = next((t for t in obs.tool_menu if t.tool_id == action.tool_id), None)
        if not tool:
            return False
        return (
            tool.estimated_latency_ms_p50 >= self.settings.big_model_latency_ms
            or tool.estimated_cost_usd_per_call >= self.settings.big_model_cost_usd
        )

    def run_episode(self, obs: RouterObservation, actions: List[RouterAction]) -> RouterEpisodeMetrics:
        total_latency = 0.0
        total_cost = 0.0
        violations = 0
        big_calls = 0

        for act in actions:
            if act.action_type == RouterActionType.ANSWER:
                # treat answer as finalization step; zero tool cost.
                pass
            else:
                tool = next((t for t in obs.tool_menu if t.tool_id == act.tool_id), None)
                if tool:
                    total_latency += tool.estimated_latency_ms_p50
                    total_cost += tool.estimated_cost_usd_per_call
                else:
                    violations += 1
                if self.is_big_model(act, obs):
                    big_calls += 1

        # Simplified success heuristic: if final action is ANSWER, count as success.
        success = 1 if actions and actions[-1].action_type == RouterActionType.ANSWER else 0
        metrics = RouterEpisodeMetrics(
            success=success,
            latency_ms=total_latency,
            cost_usd=total_cost,
            violations=violations,
            big_model_calls=big_calls,
            num_steps=len(actions),
            reward=0.0,
        )
        metrics.reward = compute_reward(metrics, self.settings)
        return metrics

