# Conversation Status (PPA Multi-Intent)

## Current Assets
- **KG (`examples/synthetic_kg.json`)**: 27 consumer intents covering disputes/refunds, payments (add card/bank link/send/checkout/PayLater/transfers/withdrawals), subscriptions, account/security, profile updates. Slots and clarifications are aligned per intent (optional additions made for SendMoney.Issue, Unauthorized, TransferToBank.Failed).
- **Patterns (`examples/multi_intent_patterns.json`)**: ~20 curated patterns (two-intent for generation; one tri-intent defined) with `component_intents`, `description`, `example_user_utterances`, `order_hint`, `shared_slots`.
- **Traces**:
  - `examples/multu_turn_traces_1.jsonl` (500 multi-intent traces)
  - `examples/multi_turn_traces.jsonl` (500 multi-intent traces)
  - Structured reasoning/tool_plan, intent-aligned tool calls, balanced coverage across 19 pairs; tri-intent excluded by default.

## Generators
- **Pattern orchestrator**: `scripts/generate_pattern_traces.py`
  - Use patterns to emit multi-intent traces.
  - Flags: `--count`, `--workers`, `--log-every`, `--use-llm`, `--llm-model`, `--tri-ratio` (fraction of tri-intent patterns; default 0).
  - Example:  
    `python3 scripts/generate_pattern_traces.py --count 500 --workers 4 --log-every 50 --use-llm --llm-model gpt-5.1-2025-11-13 --output examples/multi_turn_traces.jsonl`
    (Add `--tri-ratio 0.05` to include tri-intent.)
- **Mixed generator**: `examples/generate_multi_turn_conversations.py` (single + random multi; older).

## Schema
- See `chat_template_overview.md` for the chat JSON schema (reasoning stages, tool_plan steps, tool_calls, canonical flow, validation checklist).

## Coverage Snapshot
- 1,000 multi-intent traces (19 two-intent patterns evenly represented).
- Domain: disputes/refunds/chargeback, payments/funding (add card/bank link/send/checkout/PayLater/transfers/withdrawals), subscriptions, account/security/profile updates.
- Tri-intent pattern defined but excluded from generation unless `--tri-ratio` > 0.

## Optional Next Steps
- If desired, include tri-intent generation (enable `--tri-ratio`).
- Further tune diagnosis/resolution phrasing for recurring-charge and PayLater decline/limit stories.
- Keep pattern metadata consistent (order_hint/shared_slots) for any new patterns.
