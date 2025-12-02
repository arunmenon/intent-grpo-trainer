"""
Run plan-aware GRPO training with TRL using plan rewards.

This is a scaffold: it wires plan rewards into TRL. For production, replace the
demo dataset and completion parser with your own plan trajectories.

Usage (demo dataset):
  python scripts/run_plan_trl.py --model-id gpt2 --steps 5 --demo

Usage (custom dataset):
  python scripts/run_plan_trl.py --model-id gpt2 --dataset /path/to/plan_dataset.jsonl --steps 50

Dataset schema (JSONL):
- "prompt": str
- "gold_plan_steps": list of dicts, each with at least {"node_id": str, "gold_slots": {...}}
- optional "plan_completed": bool
- any additional metadata your completion parser needs

Completions are expected to be JSON with a "steps" list; see examples/plan_trl_integration.py
for the default parser. Override if your model emits a different format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


def require_deps():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import datasets  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        missing = str(exc).split()[-1].strip("'\"")
        raise SystemExit(
            f"Missing dependency ({missing}). Install via: pip install transformers trl datasets"
        ) from exc


require_deps()

from datasets import Dataset

from examples.plan_trl_integration import build_plan_trl_grpo_trainer, default_parse_completion
from grpo_intent.plan_schema import PlanBranch, PlanNode, PlanSpec, ToolSpec


def build_plan_spec() -> PlanSpec:
    """Same simple refund plan used in the scoring/demo scripts."""

    def has_account(state):
        return state.get("has_account", False)

    def refund_effect(before, after):
        return bool(after.get("refund_applied") and not before.get("refund_applied"))

    def full_refund_guard(state):
        return bool(state.get("eligible_full"))

    def partial_refund_guard(state):
        return not state.get("eligible_full", False)

    def tool_validator(call):
        return float(call.get("amount", 0) > 0)

    decide = PlanNode(
        node_id="decide_refund",
        required_slots=["order_id"],
        preconditions=[has_account],
        branches=[
            PlanBranch(branch_id="full", guard=full_refund_guard),
            PlanBranch(branch_id="partial", guard=partial_refund_guard),
        ],
    )

    refund = PlanNode(
        node_id="apply_refund",
        required_slots=["amount"],
        preconditions=[has_account],
        effects=[refund_effect],
        tool_spec=ToolSpec(name="refund_api", validator=tool_validator),
    )

    return PlanSpec(nodes={decide.node_id: decide, refund.node_id: refund}, start_node=decide.node_id)


def demo_dataset() -> Dataset:
    """One-row dataset with gold plan steps and a friendly prompt."""
    rows = [
        {
            "prompt": "Handle a refund for order ORD123 (customer is eligible for full refund).",
            "gold_plan_steps": [
                {"node_id": "decide_refund", "gold_slots": {"order_id": "ORD123"}},
                {"node_id": "apply_refund", "gold_slots": {"amount": "100.00"}},
            ],
            "plan_completed": True,
        }
    ]
    return Dataset.from_list(rows)


def load_plan_dataset(path: Path) -> Dataset:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows loaded from {path}")
    return Dataset.from_list(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with plan-aware rewards.")
    parser.add_argument("--model-id", required=True, help="HF model name or local path.")
    parser.add_argument("--dataset", type=Path, help="Path to JSONL plan dataset.")
    parser.add_argument("--steps", type=int, default=10, help="Max training steps.")
    parser.add_argument("--batch-size", type=int, default=1, help="per_device_train_batch_size.")
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient_accumulation_steps.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--initial-k", type=int, default=4, help="num_generations for GRPO.")
    parser.add_argument("--max-completion-length", type=int, default=128, help="max tokens per generation.")
    parser.add_argument("--demo", action="store_true", help="Use built-in demo dataset instead of loading JSONL.")
    parser.add_argument("--no-cuda-check", action="store_true", help="Skip CUDA availability check.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.no_cuda_check:
        try:
            import torch
            if not torch.cuda.is_available():
                print("[warn] CUDA not available; training will run on CPU unless model forces otherwise.")
        except Exception as exc:
            print(f"[warn] CUDA check skipped due to error: {exc}")

    if args.demo:
        ds = demo_dataset()
    else:
        if not args.dataset:
            raise SystemExit("Provide --dataset or use --demo")
        if not args.dataset.exists():
            raise SystemExit(f"Dataset not found: {args.dataset}")
        ds = load_plan_dataset(args.dataset)

    plan_spec = build_plan_spec()
    trainer = build_plan_trl_grpo_trainer(
        model_name=args.model_id,
        train_dataset=ds,
        plan_spec=plan_spec,
        parse_completion_fn=default_parse_completion,
        initial_k=args.initial_k,
        max_completion_length=args.max_completion_length,
        lr=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        report_to=None,
    )

    trainer.train(max_steps=args.steps)
    print(f"Completed {args.steps} GRPO steps with model={args.model_id}")


if __name__ == "__main__":
    main()
