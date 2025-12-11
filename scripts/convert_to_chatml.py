"""
Convert SFT samples (from split_for_sft.py) into ChatML-formatted text lines for Qwen-style models.

Each output line is a JSON object:
- conversation_id
- assistant_turn_index
- metadata
- stage
- chatml: the ChatML-formatted string (no tokenization)

Usage:
  python3 scripts/convert_to_chatml.py \
    --input examples/multi_turn_traces_sft.jsonl \
    --output examples/multi_turn_traces_chatml.jsonl

Notes:
- Uses the existing message schema; flattens tool messages into assistant text: "[tool:{tool_call_id}] {content}".
- Assistant messages are serialized as JSON strings (content + reasoning + tool_plan + tool_calls) to preserve structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    for m in sample.get("history", []):
        role = m.get("role")
        if role in ("system", "user"):
            msgs.append({"role": role, "content": m.get("content", "")})
        elif role == "assistant":
            # Preserve full assistant message structure
            msgs.append({"role": "assistant", "content": json.dumps(m, ensure_ascii=False)})
        elif role == "tool":
            tool_txt = f"[tool:{m.get('tool_call_id','')}] {m.get('content','')}"
            msgs.append({"role": "assistant", "content": tool_txt})
    target = sample.get("target", {})
    msgs.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return msgs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SFT samples to ChatML text.")
    parser.add_argument("--input", type=Path, required=True, help="Input SFT JSONL (from split_for_sft.py).")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with ChatML strings.")
    args = parser.parse_args()

    out_rows: List[Dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            chatml_messages = build_messages(sample)
            out_rows.append(
                {
                    "conversation_id": sample.get("conversation_id"),
                    "assistant_turn_index": sample.get("assistant_turn_index"),
                    "metadata": sample.get("metadata", {}),
                    "stage": sample.get("stage"),
                    "messages": chatml_messages,
                    # ChatML string is left to tokenizer.apply_chat_template at training time.
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_rows)} chatml-ready rows to {args.output}")


if __name__ == "__main__":
    main()
