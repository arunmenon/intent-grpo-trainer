"""
Split conversation JSONL traces into progressive SFT samples.

For each assistant turn, emit a sample with:
- conversation_id
- assistant_turn_index (0-based within the conversation)
- metadata (intents, decision_category, difficulty, copied from source)
- history: all messages BEFORE the assistant turn
- target: the assistant message (content + reasoning + tool_plan + tool_calls, if any)
- stage: convenience field from target.reasoning.stage (if present)

Usage:
  python3 scripts/split_for_sft.py --input examples/multi_turn_traces.jsonl --output examples/multi_turn_traces_sft.jsonl

Notes:
- Works for single- and multi-intent conversations.
- Keeps messages exactly as in source for history/target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_conversations(paths: List[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def split_conversation(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = conv.get("messages", [])
    metadata = conv.get("metadata", {})
    conv_id = conv.get("conversation_id", "")
    samples: List[Dict[str, Any]] = []
    assistant_idx = 0
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        history = messages[:idx]
        target = msg
        stage = None
        reasoning = target.get("reasoning")
        if isinstance(reasoning, dict):
            stage = reasoning.get("stage")
        samples.append(
            {
                "conversation_id": conv_id,
                "assistant_turn_index": assistant_idx,
                "metadata": metadata,
                "history": history,
                "target": target,
                "stage": stage,
            }
        )
        assistant_idx += 1
    return samples


def save_jsonl(samples: Iterable[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split conversation JSONL into progressive SFT samples.")
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Input JSONL conversation file(s).")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path for SFT samples.")
    parser.add_argument("--log-every", type=int, default=0, help="Print progress every N conversations (0 to disable).")
    args = parser.parse_args()

    samples: List[Dict[str, Any]] = []
    for i, conv in enumerate(load_conversations(args.input), 1):
        samples.extend(split_conversation(conv))
        if args.log_every and i % args.log_every == 0:
            print(f"[split] processed {i} conversations...", flush=True)

    save_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} SFT samples to {args.output}")


if __name__ == "__main__":
    main()
