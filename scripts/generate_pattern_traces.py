"""
Generate multi-intent conversation traces driven by the curated pattern list.

This orchestrator:
- Loads intents from examples/synthetic_kg.json and patterns from examples/multi_intent_patterns.json
- For each pattern, emits conversations using conversation_template_multi from the generator
- Evenly cycles through patterns until the requested count is reached

Usage (template mode):
  python3 scripts/generate_pattern_traces.py --count 500 --output examples/mock_pattern_traces.jsonl

Usage (LLM mode, OpenAI-compatible via liteLLM):
  export LITELLM_API_KEY=...
  python3 scripts/generate_pattern_traces.py --count 500 --use-llm --llm-model gpt-4o-mini --output examples/mock_pattern_traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import concurrent.futures
from pathlib import Path
from typing import Dict, List

# Allow imports from examples/
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
EXAMPLES_DIR = ROOT / "examples"
sys.path.append(str(EXAMPLES_DIR))

from synthetic_kg import load_multi_intent_patterns, load_synthetic_intents  # type: ignore
from generate_multi_turn_conversations import (  # type: ignore
    LLMGenerator,
    conversation_template_multi,
)


def load_intent_index() -> Dict[str, dict]:
    intents = load_synthetic_intents()
    return {i["intent_id"]: i for i in intents}


def normalize_patterns(patterns: List[dict], intent_index: Dict[str, dict]) -> List[dict]:
    """Filter patterns to those with exactly two or three intents and all intents present in the KG."""
    out = []
    for p in patterns:
        intents = p.get("component_intents", [])
        if len(intents) not in (2, 3):
            continue
        if any(i not in intent_index for i in intents):
            continue
        out.append(p)
    return out


def _build_conversation(idx: int, pattern: dict, intent_index: Dict[str, dict], llm: LLMGenerator | None) -> dict:
    comp = pattern["component_intents"]
    if len(comp) == 2:
        intent_a = intent_index[comp[0]]
        intent_b = intent_index[comp[1]]
        return conversation_template_multi(intent_a, intent_b, idx, llm)
    intent_a = intent_index[comp[0]]
    intent_b = intent_index[comp[1]]
    intent_c = intent_index[comp[2]]
    return conversation_template_tri(intent_a, intent_b, intent_c, idx, llm)


def cycle_patterns(
    patterns: List[dict],
    intent_index: Dict[str, dict],
    count: int,
    llm: LLMGenerator | None,
    log_every: int = 0,
    workers: int = 1,
    tri_ratio: float = 0.0,
) -> List[dict]:
    """Evenly cycle through patterns (two-intent only) and build conversations until reaching count."""
    if not patterns:
        return []
    two_intent = [p for p in patterns if len(p.get("component_intents", [])) == 2]
    tri_intent = [p for p in patterns if len(p.get("component_intents", [])) == 3]

    tasks: List[tuple[int, dict]] = []
    if tri_intent and tri_ratio > 0:
        n_tri = min(len(tri_intent), int(count * tri_ratio))
    else:
        n_tri = 0
    n_two = count - n_tri

    if n_two > 0:
        shuffled_two = two_intent[:]
        random.shuffle(shuffled_two)
        for idx in range(n_two):
            pat = shuffled_two[idx % len(shuffled_two)]
            tasks.append((idx, pat))
    if n_tri > 0:
        shuffled_tri = tri_intent[:]
        random.shuffle(shuffled_tri)
        base_idx = n_two
        for i in range(n_tri):
            pat = shuffled_tri[i % len(shuffled_tri)]
            tasks.append((base_idx + i, pat))

    conversations: List[dict] = []

    if workers and workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_build_conversation, t_idx, pat, intent_index, llm): t_idx for t_idx, pat in tasks
            }
            for i, fut in enumerate(concurrent.futures.as_completed(future_map), 1):
                conversations.append(fut.result())
                if log_every and i % log_every == 0:
                    print(f"[gen] built {i}/{len(tasks)} conversations...", flush=True)
        conversations.sort(key=lambda c: c.get("conversation_id", ""))
    else:
        for i, (t_idx, pat) in enumerate(tasks, 1):
            conversations.append(_build_conversation(t_idx, pat, intent_index, llm))
            if log_every and i % log_every == 0:
                print(f"[gen] built {i}/{len(tasks)} conversations...", flush=True)

    return conversations


def save_jsonl(rows: List[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-intent traces using curated patterns.")
    parser.add_argument("--output", type=Path, default=ROOT / "examples" / "mock_pattern_traces.jsonl", help="Output JSONL path.")
    parser.add_argument("--count", type=int, default=500, help="Number of conversations to generate (default 500).")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--pattern-source", type=Path, default=EXAMPLES_DIR / "multi_intent_patterns.json", help="Path to multi_intent_patterns.json.")
    parser.add_argument("--use-llm", action="store_true", help="Use liteLLM for user utterances.")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model name.")
    parser.add_argument("--llm-base-url", type=str, default=None, help="Optional custom base URL for OpenAI-compatible endpoints.")
    parser.add_argument("--llm-api-key-env", type=str, default="LITELLM_API_KEY", help="Env var holding the primary API key (falls back to OPENAI_API_KEY).")
    parser.add_argument("--llm-temperature", type=float, default=0.8, help="LLM temperature.")
    parser.add_argument("--llm-max-tokens", type=int, default=80, help="Max tokens for LLM completion.")
    parser.add_argument("--log-every", type=int, default=50, help="Print progress every N conversations (0 to disable).")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads for generation (template + tool planning only; LLM is threadsafe).")
    parser.add_argument("--tri-ratio", type=float, default=0.05, help="Probability of sampling a tri-intent pattern (if any exist).")
    args = parser.parse_args()

    random.seed(args.seed)

    llm = None
    if args.use_llm:
        try:
            llm = LLMGenerator(
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
            )
            print(f"LLM generation enabled: model={args.llm_model} base_url={args.llm_base_url or 'default'}")
            if "gpt-5" in args.llm_model and args.llm_temperature != 1:
                print("Note: temperature reset to 1 for gpt-5 models to match API constraints.")
        except Exception as exc:
            raise SystemExit(f"Failed to initialize LLM generator: {exc}") from exc

    pattern_path = args.pattern_source
    if not pattern_path.exists():
        raise SystemExit(f"Pattern file not found: {pattern_path}")

    # Use loader to allow future validation; keeps in sync with examples/synthetic_kg.py
    patterns = load_multi_intent_patterns()
    intents = load_intent_index()
    patterns = normalize_patterns(patterns, intents)
    if not patterns:
        raise SystemExit("No valid patterns found (need at least two intents per pattern and all intents present).")

    conversations = cycle_patterns(patterns, intents, args.count, llm, log_every=args.log_every, workers=args.workers)
    save_jsonl(conversations, args.output)
    print(f"Wrote {len(conversations)} conversations to {args.output}")


if __name__ == "__main__":
    main()
