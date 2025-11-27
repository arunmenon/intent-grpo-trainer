"""
Convenience wrapper to run DecisionObject GRPO training on GPU.

Usage:
  python scripts/run_decision_grpo.py \
    --model-id gpt2 \
    --dataset examples/mock_rl_dataset.jsonl \
    --steps 50 \
    --batch-size 1 \
    --grad-accum 1

Flags:
  --model-id: HF model name or path.
  --dataset: JSONL file with DecisionObject fields.
  --steps: max training steps.
  --batch-size: per_device_train_batch_size.
  --grad-accum: gradient_accumulation_steps.
  --lr: learning rate (default 5e-6).
  --weight-decay: weight decay (default 0.01).
  --reward-variant: legacy or decision_object (default decision_object).

This script assumes CUDA is available; device_map="auto" will place the model on GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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

from examples.trl_integration import build_trl_grpo_trainer, load_decision_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with DecisionObject rewards.")
    parser.add_argument("--model-id", required=True, help="HF model name or local path.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to DecisionObject JSONL.")
    parser.add_argument("--steps", type=int, default=50, help="Max training steps.")
    parser.add_argument("--batch-size", type=int, default=1, help="per_device_train_batch_size.")
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient_accumulation_steps.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--reward-variant", type=str, default="decision_object", choices=["decision_object", "legacy"], help="Reward variant to use.")
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

    if not args.dataset.exists():
        raise SystemExit(f"Dataset file not found: {args.dataset}")

    ds = load_decision_dataset(str(args.dataset))
    trainer = build_trl_grpo_trainer(
        model_name=args.model_id,
        train_dataset=ds,
        initial_k=8,
        max_completion_length=64,
        lr=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        report_to=None,
        reward_variant=args.reward_variant,
    )
    trainer.train(max_steps=args.steps)
    print(f"Completed {args.steps} GRPO steps with model={args.model_id}, dataset={args.dataset}")


if __name__ == "__main__":
    main()
