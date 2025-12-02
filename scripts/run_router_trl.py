"""
Run router GRPO training (scaffold) with plan/tool-aware rewards.

Demo usage (minimal dataset):
  python scripts/run_router_trl.py --model-id gpt2 --steps 5 --demo

Dataset schema (JSONL) for custom runs:
- "prompt": str
- "tool_menu": list of tool dicts (RouterTool fields)
- optional: "intent_prior", "kg_context", "partial_trace", "budget", "environment"

Completions should be RouterAction JSON. The reward currently treats each completion
as a single-step action; extend to multi-step traces as needed.
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

from examples.router_trl_integration import build_router_grpo_trainer
from grpo_intent.router_schema import RouterTool


def demo_dataset() -> Dataset:
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
        ).__dict__,
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
        ).__dict__,
    ]
    rows = [
        {
            "request_id": "demo-1",
            "turn_id": 0,
            "prompt": "I need a refund for transaction TXN-123",
            "tool_menu": tools,
            "intent_prior": {},
            "kg_context": {},
            "partial_trace": [],
            "budget": {"max_latency_ms": 2000, "max_cost_usd": 0.02},
            "environment": {"geo": "US"},
        }
    ]
    return Dataset.from_list(rows)


def load_dataset(path: Path) -> Dataset:
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
    parser = argparse.ArgumentParser(description="Run GRPO training with router rewards.")
    parser.add_argument("--model-id", required=True, help="HF model name or local path.")
    parser.add_argument("--dataset", type=Path, help="Path to JSONL router dataset.")
    parser.add_argument("--steps", type=int, default=5, help="Max training steps.")
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
        ds = load_dataset(args.dataset)

    trainer = build_router_grpo_trainer(
        model_name=args.model_id,
        train_dataset=ds,
        initial_k=args.initial_k,
        max_completion_length=args.max_completion_length,
        lr=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        report_to=None,
    )
    trainer.train(max_steps=args.steps)
    print(f"Completed {args.steps} router GRPO steps with model={args.model_id}")


if __name__ == "__main__":
    main()
