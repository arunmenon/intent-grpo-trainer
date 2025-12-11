# 📊 Multi-Intent PPA Router – Current Status & Changes

## 1. What we’ve built so far (bird’s-eye view)

### Single-intent KG (`synthetic_kg.json`)
- 27 leaf intents for consumer PPA:
  - Disputes/Refunds: ItemNotReceived, NotAsDescribed, DamagedItem, WrongItemReceived, Unauthorized, RecurringCharge, PartialRefund, FullRefund, RefundStatus, Chargeback.Response
  - Payments/Funding: SendMoney.Issue, AddCard.Failed, BankLink.Verification, TransferToBank.Failed, Withdrawals.Delayed, Withdrawals.Reversed, Checkout.CardDeclined, PayLater.PaymentDeclined, PayLater.LimitIncrease, Subscriptions.CancelRecurring, Subscriptions.ChangePlan
  - Account/Security: Access.LockedOut, Access.Limited, Profile.UpdateName, Profile.UpdateAddress, Security.TwoFactorReset, Security.SuspectedCompromise
- Each intent has required/optional slots + matching clarification_templates.

### Multi-intent patterns (`multi_intent_patterns.json`)
- ~20 curated two-intent patterns (tri-intent defined but currently off for generation) with:
  - component_intents
  - description
  - example_user_utterances
  - order_hint (processing order)
  - shared_slots (conversation-level slots to collect once)

### Multi-turn traces (`multi_turn_traces*.jsonl`)
- ~1,000 multi-intent conversations with:
  - Structured reasoning traces
  - Explicit tool plans
  - Intent-aligned diagnosis/resolution texts

## 2. KG & pattern changes (what we essentially did)
- Expanded intent coverage (more item issues, refund status, subscription change, account risk/access, profile updates).
- Slot/clarification consistency per intent.
- Pattern layer: added order_hint/shared_slots to guide plan ordering and slot reuse; patterns span dispute↔refund, dispute↔subscription, unauthorized/security↔access, funding/withdrawals↔payments/account, PayLater decline↔limit, address↔checkout decline.
- Nice-to-have slot tweaks (now included): optional txn_id on TransferToBank.Failed; optional bank_name/error_message on SendMoney.Issue; optional login_email on Unauthorized.

## 3. Chat template format (how to read a trace)
- Top-level: conversation_id, metadata (intents, decision_category, difficulty), messages.
- Roles: system/user (content only); assistant (content + reasoning + tool_plan + optional tool_calls); tool (tool_call_id + content).
- Reasoning (assistant): {stage ∈ {clarify_slots, diagnosis, resolution, final_summary}, focus_intents, summary}.
- Tool plan: {strategy, steps[{step_id, tool, task ∈ {collect_slots, diagnose_intent, resolve_intent, summarize_outcome}, intent_ids, phase, slots_used, order, parallel_group?, depends_on?}]}.
- Flow per convo: clarify_slots → user fills → diagnosis (per intent) → resolution (per intent) → final_summary (show_to_user).

## 4. Coverage snapshot (current state)
- 1,000 multi-intent traces (two 500-trace runs).
- 19 two-intent patterns represented, evenly distributed (~26–27 each); tri-intent excluded by generator for now.
- Domain coverage (consumer PPA): disputes/refunds/chargeback; payments (add card/bank link/send money/checkout/PayLater/transfers/withdrawals); subscriptions; account/security (locked out/limited/2FA reset/compromise/profile updates).

## 5. If you want to extend from here
- KG polish: ensure optional slots on TransferToBank.Failed (txn_id), SendMoney.Issue (bank_name/error_message), Unauthorized (login_email).
- Patterns: ensure every pattern has order_hint + shared_slots; use shared_slots as conversation-level bridging slots.
- Text coherence: tune diag/resolution wording for recurring-charge (billing phrasing), PayLater decline vs limit increase, and Withdrawals.Delayed if you don’t want “reversed” outcomes.
- Tri-intent strategy: keep it off (eval-only) or enable a small tri_ratio (5–10%) with a tri-intent template.

## 6. Handy commands
- Pattern-driven generation (2-intent only):  
  `python3 scripts/generate_pattern_traces.py --count 500 --workers 4 --log-every 50 --use-llm --llm-model gpt-5.1-2025-11-13 --output examples/multi_turn_traces.jsonl`
- Enable tri-intent sampling (e.g., 5%): add `--tri-ratio 0.05`
- Mixed single/multi generator (older, random intents):  
  `python3 examples/generate_multi_turn_conversations.py --count N --multi-ratio 0.35 [...]`
