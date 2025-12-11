# CONVERSATION_SCHEMA (Chat Template Overview)

This file defines the JSON schema for multi-turn PPA conversations used to train the router + planner + tool-usage stack.

---

## 1. Top-Level Conversation Object

Each conversation is a single JSON object:

```jsonc
{
  "conversation_id": "conv_multi_0000",
  "metadata": {
    "intents": [
      "PPA.Disputes.RecurringCharge",
      "PPA.Payments.Subscriptions.ChangePlan"
    ],
    "decision_category": "multi_intent",
    "difficulty": "complex"
  },
  "messages": [
    // list of messages (system, user, assistant, tool)
  ]
}
```

### 1.1 `conversation_id` (string, required)
- Unique ID for the conversation (e.g., `"conv_multi_0000"`).

### 1.2 `metadata` (object, required)
- `intents` (array of strings): intent_id(s) from the KG (one for single-intent, ≥2 for multi-intent).
- `decision_category` (string): `"single_intent"` or `"multi_intent"`.
- `difficulty` (string): `"simple"` or `"complex"`.

### 1.3 `messages` (array, required)
- Ordered list of messages. `role` ∈ {`system`, `user`, `assistant`, `tool`}.

---

## 2. Message Format by Role

### 2.1 System Message
```json
{ "role": "system", "content": "You are a routing assistant..." }
```
No reasoning/tool_plan/tool_calls.

### 2.2 User Message
```json
{ "role": "user", "content": "I noticed an unexpected charge..." }
```
No reasoning/tool_plan/tool_calls.

### 2.3 Assistant Message
- `role`: `"assistant"`
- `content`: user-facing text (optional but recommended)
- `reasoning`: structured thought bubble (required)
- `tool_plan`: structured tool-use plan (required)
- `tool_calls`: actual function/tool calls (optional, only when executing a step)

Example (clarify turn):
```json
{
  "role": "assistant",
  "content": "I’ll handle both intents. I need a couple details...",
  "reasoning": {
    "stage": "clarify_slots",
    "focus_intents": ["PPA.Disputes.RecurringCharge", "PPA.Payments.Subscriptions.ChangePlan"],
    "summary": "Confirm both intents and collect missing slots before tool calls."
  },
  "tool_plan": {
    "strategy": "sequential",
    "steps": [
      {
        "step_id": "collect_0_plan",
        "tool": "none",
        "task": "collect_slots",
        "intent_ids": [
          "PPA.Disputes.RecurringCharge",
          "PPA.Payments.Subscriptions.ChangePlan"
        ],
        "phase": "clarify_slots",
        "slots_used": [],
        "order": 0,
        "parallel_group": "collect"
      },
      {
        "step_id": "diag_a_0_plan",
        "tool": "call_commerce_agent",
        "task": "diagnose_intent",
        "intent_ids": ["PPA.Disputes.RecurringCharge"],
        "phase": "diagnosis",
        "slots_used": ["date", "merchant", "reason", "txn_id"],
        "order": 1,
        "parallel_group": "diag",
        "depends_on": ["collect_0_plan"]
      }
    ]
  }
}
```

### 2.4 Tool Message
```json
{
  "role": "tool",
  "tool_call_id": "diag_a_0",
  "content": "Diagnosis: ... Slots={...}"
}
```
No reasoning/tool_plan.

---

## 3. Assistant Reasoning Schema
Every assistant message must include:
```json
"reasoning": {
  "stage": "clarify_slots" | "diagnosis" | "resolution" | "final_summary",
  "focus_intents": ["PPA.Payments.Withdrawals.Delayed", "..."],
  "summary": "Short explanation of why this turn exists."
}
```

---

## 4. Assistant Tool Plan Schema
Every assistant message must include:
```json
"tool_plan": {
  "strategy": "sequential" | "parallel",
  "steps": [
    {
      "step_id": "diag_a_0_plan",
      "tool": "call_ppa_agent" | "call_commerce_agent" | "show_to_user" | "none",
      "task": "collect_slots" | "diagnose_intent" | "resolve_intent" | "summarize_outcome",
      "intent_ids": ["..."],
      "phase": "clarify_slots" | "diagnosis" | "resolution" | "final_summary",
      "slots_used": ["txn_id", "merchant", "date"],
      "order": 1,
      "parallel_group": "diag",
      "depends_on": ["collect_0_plan"]
    }
  ]
}
```
- `strategy`: execution strategy (usually `"sequential"`).
- `steps`: ordered list of planned tool steps.

---

## 5. Tool Calls
Assistant turns that execute a step include `tool_calls`:
```json
"tool_calls": [
  {
    "id": "diag_a_0",
    "type": "function",
    "function": {
      "name": "call_commerce_agent",
      "arguments": "{\"task\": \"diagnose_intent\", \"intent_id\": \"...\", \"slots\": {...}}"
    }
  }
]
```
Tool replies follow with `role = "tool"` and matching `tool_call_id`.

---

## 6. Canonical Flow
- **Single-intent:** clarify_slots → user fills → diagnosis → resolution → final_summary.
- **Multi-intent (2 intents):** same flow, but one diagnosis + one resolution turn per intent; final summary covers all intents. (Tri-intent uses the same pattern with three branches.)

---

## 7. Validation Checklist
- Assistant turns always have `reasoning` + `tool_plan`.
- `focus_intents` ⊆ `metadata.intents`.
- `tool_plan.steps.phase` aligns with `reasoning.stage`.
- `tool_calls` have matching `tool` messages (via `tool_call_id`).
- `intent_ids` and `slots_used` are valid against the KG/patterns.
