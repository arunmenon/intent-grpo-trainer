"""
Schema and type definitions for the router (step-wise policy).

Keep these JSON-safe and lenient to plug into TRL reward parsing and demos.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class RouterActionType(str, Enum):
    CALL_TOOL = "CALL_TOOL"
    EXECUTE_PLAN = "EXECUTE_PLAN"
    ANSWER = "ANSWER"


@dataclass
class RouterTool:
    """Tool descriptor passed in the tool menu."""

    tool_id: str
    kind: str  # CLASSIFIER | API | LLM | PLAN_EXECUTOR | OTHER
    domain_tags: Sequence[str] = field(default_factory=list)
    estimated_latency_ms_p50: float = 0.0
    estimated_cost_usd_per_call: float = 0.0
    reliability: float = 1.0
    schema_in: Dict[str, Any] = field(default_factory=dict)
    schema_out: Dict[str, Any] = field(default_factory=dict)
    kg_links: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentPrior:
    """ModernBERT-style intent hierarchy probabilities."""

    domain: Sequence[Dict[str, Any]] = field(default_factory=list)
    pillar: Sequence[Dict[str, Any]] = field(default_factory=list)
    subpillar: Sequence[Dict[str, Any]] = field(default_factory=list)
    intent: Sequence[Dict[str, Any]] = field(default_factory=list)
    subintent: Sequence[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RouterObservation:
    """
    Single decision-step observation for the router policy.
    """

    request_id: str
    turn_id: int
    query: str
    conversation_history: List[Dict[str, str]]
    intent_prior: IntentPrior = field(default_factory=IntentPrior)
    kg_context: Dict[str, Any] = field(default_factory=dict)
    tool_menu: List[RouterTool] = field(default_factory=list)
    partial_trace: List[Dict[str, Any]] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterAction:
    """
    Router decision output for one step.
    """

    decision_id: str
    step_index: int
    action_type: RouterActionType
    tool_id: Optional[str] = None
    plan_id: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None
    finish_reason: str = "continue"  # "continue" | "done"
    logprobs: Any = None
    router_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    """
    Result of executing a RouterAction; used to build partial_trace in obs.
    """

    step_index: int
    action: RouterAction
    result_summary: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    status: str = "OK"  # "OK" | "ERROR"
    violations: List[str] = field(default_factory=list)

