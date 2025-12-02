"""
Lightweight Plan DAG and runtime record types for multi-turn agent rewards.

These classes are intentionally minimal and callable-friendly so you can plug in
your own precondition/effect/guard/tool validators without coupling to a
specific tool runtime or data model.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

State = Mapping[str, Any]
ConditionCheck = Callable[[State], bool]
EffectCheck = Callable[[State, State], bool]
ToolValidator = Callable[[Any], float]
BranchGuard = Callable[[State], bool]


@dataclass
class ToolSpec:
    """Declarative wrapper for tool validation."""

    name: str
    validator: Optional[ToolValidator] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanBranch:
    """Branch choice for a node (e.g., Full vs Partial refund)."""

    branch_id: str
    guard: BranchGuard


@dataclass
class PlanNode:
    """Single plan node with slots, preconditions, effects, and branch guards."""

    node_id: str
    required_slots: Sequence[str] = field(default_factory=list)
    optional_slots: Sequence[str] = field(default_factory=list)
    preconditions: Sequence[ConditionCheck] = field(default_factory=list)
    effects: Sequence[EffectCheck] = field(default_factory=list)
    branches: Sequence[PlanBranch] = field(default_factory=list)
    tool_spec: Optional[ToolSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanSpec:
    """
    Minimal DAG-like plan container keyed by node_id.

    - nodes: mapping of node_id -> PlanNode
    - start_node: optional convenience pointer to the entry node
    """

    nodes: Dict[str, PlanNode]
    start_node: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def node(self, node_id: str) -> PlanNode:
        """Lookup helper with a clearer error message."""
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown node_id '{node_id}' in PlanSpec") from exc

    def is_terminal(self, node_id: str) -> bool:
        """A node is terminal if it has no outgoing branches."""
        return not self.node(node_id).branches
