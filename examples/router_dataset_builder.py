"""
Toy ToolScale-style router dataset generator (JSONL).

Outputs records suitable for the router TRL scaffold:
{
  "request_id": "uuid",
  "turn_id": 0,
  "prompt": "...",
  "tool_menu": [ {RouterTool fields}, ... ],
  "gold_actions": [ {RouterAction JSON}, ... ],
  "budget": {...},
  "environment": {...}
}

Usage:
  python examples/router_dataset_builder.py --output examples/router_demo.jsonl --count 200 --seed 42 --include-decoys
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import List

from grpo_intent.router_schema import RouterAction, RouterActionType, RouterTool


def build_tool_inventory(include_decoys: bool) -> List[RouterTool]:
    tools = [
        RouterTool(
            tool_id="ModernBERT.IntentHierarchy",
            kind="CLASSIFIER",
            domain_tags=["PPA"],
            estimated_latency_ms_p50=30,
            estimated_cost_usd_per_call=0.0001,
            reliability=0.99,
            schema_in={"type": "text"},
            schema_out={"type": "intent_hierarchy"},
            kg_links=[],
        ),
        RouterTool(
            tool_id="LedgerAPI.GetTransaction",
            kind="API",
            domain_tags=["PPA", "Refunds"],
            estimated_latency_ms_p50=120,
            estimated_cost_usd_per_call=0.00005,
            reliability=0.999,
            schema_in={"txn_id": "string"},
            schema_out={"txn": "TransactionRecord"},
            kg_links=["RefundPayment", "ChargeDispute"],
        ),
        RouterTool(
            tool_id="RefundAPI.CreateRefund",
            kind="API",
            domain_tags=["PPA", "Refunds"],
            estimated_latency_ms_p50=150,
            estimated_cost_usd_per_call=0.0001,
            reliability=0.999,
            schema_in={"txn_id": "string", "amount": "money", "currency": "string"},
            schema_out={"refund_id": "string", "status": "string"},
            kg_links=["RefundPayment.FullRefund", "RefundPayment.PartialRefund"],
        ),
        RouterTool(
            tool_id="PPA-Reasoner-14B",
            kind="LLM",
            domain_tags=["PPA", "Commerce"],
            estimated_latency_ms_p50=450,
            estimated_cost_usd_per_call=0.003,
            reliability=0.97,
            schema_in={"type": "chat"},
            schema_out={"type": "chat"},
            kg_links=[],
        ),
    ]

    if include_decoys:
        tools.extend(
            [
                RouterTool(
                    tool_id="MerchantOnboarding.Checklist",
                    kind="API",
                    domain_tags=["Merchant"],
                    estimated_latency_ms_p50=80,
                    estimated_cost_usd_per_call=0.00005,
                    reliability=0.95,
                    schema_in={"mid": "string"},
                    schema_out={"status": "string"},
                    kg_links=["MerchantOnboarding"],
                ),
                RouterTool(
                    tool_id="PPA-Reasoner-70B",
                    kind="LLM",
                    domain_tags=["PPA", "Commerce"],
                    estimated_latency_ms_p50=900,
                    estimated_cost_usd_per_call=0.009,
                    reliability=0.98,
                    schema_in={"type": "chat"},
                    schema_out={"type": "chat"},
                    kg_links=[],
                ),
            ]
        )
    return tools


def build_refund_trace(txn_id: str, amount: str, currency: str) -> List[RouterAction]:
    return [
        RouterAction(
            decision_id=str(uuid.uuid4()),
            step_index=0,
            action_type=RouterActionType.CALL_TOOL,
            tool_id="LedgerAPI.GetTransaction",
            args={"txn_id": txn_id},
            finish_reason="continue",
        ),
        RouterAction(
            decision_id=str(uuid.uuid4()),
            step_index=1,
            action_type=RouterActionType.CALL_TOOL,
            tool_id="RefundAPI.CreateRefund",
            args={"txn_id": txn_id, "amount": amount, "currency": currency},
            finish_reason="continue",
        ),
        RouterAction(
            decision_id=str(uuid.uuid4()),
            step_index=2,
            action_type=RouterActionType.ANSWER,
            answer=f"I've created a refund of {amount} {currency} for transaction {txn_id}.",
            finish_reason="done",
        ),
    ]


def sample_prompt(txn_id: str, amount: str, currency: str) -> str:
    templates = [
        f"I need a refund of {amount} {currency} for transaction {txn_id}.",
        f"My charge {txn_id} looks wrong, please refund {amount} {currency}.",
        f"Can you issue a partial refund ({amount} {currency}) on txn {txn_id}?",
    ]
    return random.choice(templates)


def action_to_dict(action: RouterAction) -> dict:
    data = action.__dict__.copy()
    data["action_type"] = action.action_type.value
    return data


def tool_to_dict(tool: RouterTool) -> dict:
    return tool.__dict__.copy()


def generate_dataset(count: int, include_decoys: bool, seed: int) -> List[dict]:
    random.seed(seed)
    tools = build_tool_inventory(include_decoys)
    records: List[dict] = []
    for idx in range(count):
        txn_id = f"TXN-{1000 + idx}"
        amount = random.choice(["10.00", "15.00", "25.00"])
        currency = random.choice(["USD", "EUR"])
        prompt = sample_prompt(txn_id, amount, currency)
        gold_actions = build_refund_trace(txn_id, amount, currency)
        record = {
            "id": f"router.demo.{idx:05d}",
            "request_id": str(uuid.uuid4()),
            "turn_id": 0,
            "prompt": prompt,
            "tool_menu": [tool_to_dict(t) for t in tools],
            "gold_actions": [action_to_dict(a) for a in gold_actions],
            "budget": {"max_latency_ms": 2000, "max_cost_usd": 0.02},
            "environment": {"geo": "US"},
        }
        records.append(record)
    return records


def write_jsonl(records: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a toy router dataset (JSONL).")
    parser.add_argument("--output", type=Path, required=True, help="Path to write JSONL.")
    parser.add_argument("--count", type=int, default=200, help="Number of records to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--include-decoys", action="store_true", help="Include decoy tools in the menu.")
    return parser.parse_args()


def main():
    args = parse_args()
    records = generate_dataset(count=args.count, include_decoys=args.include_decoys, seed=args.seed)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} router examples to {args.output}")


if __name__ == "__main__":
    main()
