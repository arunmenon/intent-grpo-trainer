"""
Generate synthetic multi-turn conversations (5–15 turns) using the mock KG.

Outputs JSONL where each line is:
- conversation_id: unique string
- metadata: intents, decision_category (single_intent | multi_intent), difficulty
- messages: OpenAI-chat-compatible messages with tool calls/results

Examples:
  python examples/generate_multi_turn_conversations.py --count 3 --output examples/mock_traces.jsonl
  python examples/generate_multi_turn_conversations.py --count 3 --use-llm --llm-model gpt-4o-mini --output examples/mock_traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from generate_mock_rl_dataset import pick_slot_value
from synthetic_kg import SYNTHETIC_INTENTS

SYSTEM_PROMPT = (
    "You are a routing assistant. Use tools (`call_ppa_agent`, `call_commerce_agent`, `show_to_user`). "
    "Be concise, ask for missing details, and summarize outcomes."
)

# Simple persona/trait vectors. Keep lightweight and human-readable.
PERSONAS: Dict[str, Dict[str, object]] = {
    "concise_support": {
        "description": "Concise PPA consumer support agent; balances empathy with brevity.",
        "traits": {"tone": "concise", "verbosity": "low", "empathy": "medium", "formality": "medium", "domain": "PPA consumer support"},
        "scenario": "default",
    },
    "reassuring_guide": {
        "description": "Warm, reassuring guide; adds brief reassurance while staying on-task.",
        "traits": {"tone": "reassuring", "verbosity": "medium", "empathy": "high", "formality": "low", "domain": "PPA consumer support"},
        "scenario": "anxious user",
    },
    "strict_ops": {
        "description": "Strict operations style; direct instructions, minimal small talk.",
        "traits": {"tone": "direct", "verbosity": "low", "empathy": "low", "formality": "high", "domain": "PPA compliance/ops"},
        "scenario": "policy-focused",
    },
}

# Lightweight phrasing helpers.
SLOT_TEMPLATES: Dict[str, str] = {
    "amount": "It was for {value}.",
    "currency": "Currency was {value}.",
    "txn_id": "Transaction ID {value}.",
    "bank_name": "Bank: {value}.",
    "receiver": "Receiver: {value}.",
    "reason": "Reason: {value}.",
    "date": "This happened {value}.",
    "device": "I was on {value}.",
    "merchant": "Merchant: {value}.",
    "seller": "Seller: {value}.",
    "login_email": "Email: {value}.",
    "error_message": "Error message: {value}.",
    "case_id": "Case ID: {value}.",
    "expected_delivery_date": "Expected delivery: {value}.",
    "tracking_id": "Tracking: {value}.",
    "issue_detail": "Issue: {value}.",
    "next_bill_date": "Next bill date: {value}.",
    "bank": "Bank: {value}.",
}

DIFFICULTY_BUCKETS = ["simple", "medium", "complex"]
INTENT_FAMILY_DIAG = {
    "Disputes": [
        "seller misrepresented item, eligible for refund",
        "item missing parts, refund eligible",
        "item not delivered on time, partial refund possible",
    ],
    "Refunds": [
        "refund pending settlement with issuer",
        "refund completed; awaiting bank posting",
        "refund on hold due to additional verification",
    ],
    "Payments.AddCard": [
        "issuer declined this card product for wallet use",
        "AVS/CVV check failed for this card",
        "temporary issuer lock; retry after verification",
    ],
    "Payments.TransferToBank": [
        "bank rejected transfer due to name mismatch",
        "bank transfer pending additional verification",
        "transfer blocked by bank risk rules",
    ],
    "Payments.Withdrawals": [
        "withdrawal pending bank review",
        "withdrawal reversed by bank",
        "transfer delayed due to account checks",
    ],
    "Payments.SendMoney": [
        "bank rejected the send due to insufficient verification",
        "send blocked by risk controls",
        "send failed due to insufficient funds or bank reject",
    ],
    "Payments.Checkout": [
        "card declined by issuer",
        "AVS/CVV mismatch at checkout",
        "merchant or issuer risk controls blocked the payment",
    ],
    "Payments.PayLater": [
        "Pay Later risk model declined this attempt",
        "insufficient Pay Later history for limit increase",
        "Pay Later decline due to merchant risk controls",
    ],
    "Payments.BankLink": [
        "micro-deposits verification still pending",
        "bank link blocked by incorrect credentials",
        "bank link requires MFA confirmation",
    ],
    "Account.Access": [
        "account locked after suspicious login pattern",
        "temporary limitation due to policy review",
        "access limited pending identity verification",
    ],
    "Account.Security": [
        "possible account takeover detected",
        "login anomalies from new device/location",
        "security hold applied pending user verification",
    ],
    "Subscriptions": [
        "recurring charge created by active subscription",
        "subscription cancellation pending next billing cycle",
    ],
}

INTENT_FAMILY_RESOLUTION = {
    "Disputes": [
        "refund initiated and seller notified",
        "replacement or refund offered to buyer",
        "case escalated for manual review",
    ],
    "Refunds": [
        "refund status shared with user; monitoring bank posting",
        "refund expedited and user notified",
    ],
    "Payments.AddCard": [
        "card marked unsupported; suggested alternate funding",
        "retry scheduled after issuer verification",
        "user prompted to check AVS/CVV with issuer",
    ],
    "Payments.TransferToBank": [
        "name updated and transfer retried",
        "transfer rerouted after verification",
        "escalated to bank liaison for clearance",
    ],
    "Payments.Withdrawals": [
        "withdrawal released after review",
        "reversal acknowledged; funds returned to balance",
        "expedited follow-up with bank on delay",
    ],
    "Payments.SendMoney": [
        "send retried after verification",
        "alternate funding suggested for send",
        "send escalated for manual approval",
    ],
    "Payments.Checkout": [
        "issuer decline acknowledged; suggested alternate card",
        "address/AVS updated and retry initiated",
        "security checks cleared; retry payment",
    ],
    "Payments.PayLater": [
        "limit increase partially approved",
        "limit increase denied; suggested lower amount",
        "decline upheld; asked for alternate funding",
    ],
    "Payments.BankLink": [
        "micro-deposits resent for verification",
        "bank link retried after credentials reset",
        "MFA prompted to complete bank link",
    ],
    "Account.Access": [
        "account unlocked after identity check",
        "limitation notice sent; awaiting documents",
        "password reset and session cleanup performed",
    ],
    "Account.Security": [
        "security hold applied; reset credentials",
        "sessions revoked and 2FA reset offered",
    ],
    "Subscriptions": [
        "subscription canceled and future charges stopped",
        "plan change applied; new billing takes effect next cycle",
    ],
}


@dataclass
class SlotPlan:
    present: Dict[str, str]
    missing: Dict[str, str]


def render_slot_sentences(slots: Dict[str, str]) -> List[str]:
    sentences = []
    for slot, value in slots.items():
        template = SLOT_TEMPLATES.get(slot, "{slot}: {value}.")
        sentences.append(template.format(slot=slot, value=value))
    return sentences


def assistant_message(
    content: str,
    reasoning: Dict[str, object],
    tool_calls: Optional[List[dict]] = None,
    tool_plan: Optional[Dict[str, object]] = None,
) -> dict:
    message: Dict[str, object] = {
        "role": "assistant",
        "content": content,
        "reasoning": reasoning,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if tool_plan:
        message["tool_plan"] = tool_plan
    return message


def make_reasoning(stage: str, focus_intents: List[str], summary: str) -> Dict[str, object]:
    return {"stage": stage, "focus_intents": focus_intents, "summary": summary}


def make_step(
    step_id: str,
    tool: str,
    task: str,
    intent_ids: List[str],
    phase: str,
    slots_used: List[str],
    order: int,
    parallel_group: Optional[str] = None,
    depends_on: Optional[List[str]] = None,
) -> Dict[str, object]:
    step: Dict[str, object] = {
        "step_id": step_id,
        "tool": tool,
        "task": task,
        "intent_ids": intent_ids,
        "phase": phase,
        "slots_used": slots_used,
        "order": order,
    }
    if parallel_group:
        step["parallel_group"] = parallel_group
    if depends_on:
        step["depends_on"] = depends_on
    return step


def make_tool_plan(steps: List[Dict[str, object]], strategy: str = "sequential") -> Dict[str, object]:
    sorted_steps = sorted(steps, key=lambda s: s.get("order", 0))
    return {"strategy": strategy, "steps": sorted_steps}


def apply_persona_to_system_prompt(persona: Optional[Dict[str, object]]) -> str:
    if not persona:
        return SYSTEM_PROMPT
    traits = persona.get("traits", {})
    trait_str = ", ".join(f"{k}: {v}" for k, v in traits.items()) if isinstance(traits, dict) else ""
    desc = persona.get("description", "")
    scenario = persona.get("scenario")
    parts = [SYSTEM_PROMPT]
    if desc:
        parts.append(f"Persona: {desc}.")
    if trait_str:
        parts.append(f"Traits: {trait_str}.")
    if scenario:
        parts.append(f"Scenario: {scenario}.")
    parts.append("Respond in line with these traits.")
    return " ".join(parts)


class LLMGenerator:
    """
    Optional LLM-backed utterance generator using liteLLM (OpenAI-compatible).
    Mirrors the pattern used in generate_mock_rl_dataset.py.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key_env: str = "LITELLM_API_KEY",
        fallback_api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.8,
        max_tokens: int = 80,
        drop_params: bool = True,
    ):
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise RuntimeError("liteLLM is required for LLM generation (pip install litellm)") from exc

        api_key = os.getenv(api_key_env) or os.getenv(fallback_api_key_env)
        if not api_key:
            raise RuntimeError(f"API key not found in env var {api_key_env} or {fallback_api_key_env}")

        litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url

        # Optional: drop unsupported params to avoid model-specific errors.
        litellm.drop_params = drop_params

        self.litellm = litellm
        self.model = model
        # Some GPT-5* models only accept temperature=1; auto-adjust if needed.
        if "gpt-5" in model and temperature != 1:
            self.temperature = 1
        else:
            self.temperature = temperature
        self.max_tokens = max_tokens
        self.drop_params = drop_params

    def _completion(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.litellm.completion(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            drop_params=self.drop_params,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def generate_single(
        self,
        intent_desc: str,
        present_slots: Dict[str, str],
        missing_slots: Dict[str, str],
        difficulty: str,
    ) -> str:
        sys_prompt = (
            "You craft a concise first user turn for a support chat. 1-2 sentences, natural language, no JSON. "
            "Include provided slot details naturally; do not invent missing slots."
        )
        slot_text = "; ".join(render_slot_sentences(present_slots)) if present_slots else "no slot details provided"
        missing_text = ", ".join(missing_slots.keys()) if missing_slots else "none"
        user_prompt = (
            f"Intent: {intent_desc}\n"
            f"Difficulty: {difficulty}\n"
            f"Include slot details: {slot_text}\n"
            f"Missing slots (do not mention): {missing_text}\n"
            "Return only the user utterance."
        )
        return self._completion(sys_prompt, user_prompt)

    def generate_multi(
        self,
        intent_a: str,
        intent_b: str,
        present_a: Dict[str, str],
        present_b: Dict[str, str],
        missing_a: Dict[str, str],
        missing_b: Dict[str, str],
        difficulty: str,
    ) -> str:
        sys_prompt = (
            "You craft one concise user turn covering two support issues. 1-2 sentences, natural language, no JSON. "
            "Include provided slot details naturally; do not invent missing slots."
        )
        slot_a = "; ".join(render_slot_sentences(present_a)) if present_a else "no slot details provided"
        slot_b = "; ".join(render_slot_sentences(present_b)) if present_b else "no slot details provided"
        miss_a = ", ".join(missing_a.keys()) if missing_a else "none"
        miss_b = ", ".join(missing_b.keys()) if missing_b else "none"
        user_prompt = (
            f"Intent A: {intent_a}\n"
            f"Slot details A: {slot_a}\n"
            f"Missing slots A (do not mention): {miss_a}\n"
            f"Intent B: {intent_b}\n"
            f"Slot details B: {slot_b}\n"
            f"Missing slots B (do not mention): {miss_b}\n"
            f"Difficulty: {difficulty}\n"
            "Write a single user utterance covering both intents."
        )
        return self._completion(sys_prompt, user_prompt)


def choose_slots(intent: dict, max_missing: int = 2) -> SlotPlan:
    required = list(intent.get("required_slots", []))
    optional_pool = list(intent.get("optional_slots", []))
    missing_count = min(max_missing, len(required))
    missing_slots = random.sample(required, missing_count) if missing_count else []
    present_slots = [s for s in required if s not in missing_slots]
    optional_slots = random.sample(optional_pool, k=min(1, len(optional_pool)))

    values: Dict[str, str] = {}
    for slot in required + optional_slots:
        values[slot] = pick_slot_value(slot)

    present = {slot: values[slot] for slot in present_slots + optional_slots if slot in values}
    missing = {slot: values.get(slot, slot) for slot in missing_slots}
    return SlotPlan(present=present, missing=missing)


def pick_diag(int_obj: dict) -> str:
    path = int_obj.get("path", [])
    keys = []
    if len(path) >= 3:
        keys.append(".".join(path[1:3]))
    if len(path) >= 2:
        keys.append(path[1])
    if path:
        keys.append(path[0])
    for k in keys:
        if k in INTENT_FAMILY_DIAG:
            return random.choice(INTENT_FAMILY_DIAG[k])
    return "issue detected"


def pick_res(int_obj: dict) -> str:
    path = int_obj.get("path", [])
    keys = []
    if len(path) >= 3:
        keys.append(".".join(path[1:3]))
    if len(path) >= 2:
        keys.append(path[1])
    if path:
        keys.append(path[0])
    for k in keys:
        if k in INTENT_FAMILY_RESOLUTION:
            return random.choice(INTENT_FAMILY_RESOLUTION[k])
    return "resolution applied"


def choose_intent_pair() -> Tuple[dict, dict]:
    intents = random.sample(SYNTHETIC_INTENTS, 2)
    return intents[0], intents[1]


def choose_intent_triple() -> Tuple[dict, dict, dict]:
    intents = random.sample(SYNTHETIC_INTENTS, 3)
    return intents[0], intents[1], intents[2]


def default_tool(intent_id: str) -> str:
    if intent_id.startswith("PPA.Disputes") or intent_id.startswith("PPA.Refunds"):
        return "call_commerce_agent"
    return "call_ppa_agent"


def build_user_message(intent_desc: str, present_slots: Dict[str, str], multi_hint: str | None = None) -> str:
    opener = f"I need help with {intent_desc}."
    if multi_hint:
        opener = f"{multi_hint} {opener}"
    slot_text = " ".join(render_slot_sentences(present_slots)) if present_slots else ""
    return f"{opener} {slot_text}".strip()


def build_user_followup(missing_slots: Dict[str, str]) -> str:
    if not missing_slots:
        return "Those are the details."
    parts = render_slot_sentences(missing_slots)
    return "Missing details: " + " ".join(parts)


def build_user_followup_multi(intent_a: str, missing_a: Dict[str, str], intent_b: str, missing_b: Dict[str, str]) -> str:
    sections = []
    if missing_a:
        sections.append(f"For {intent_a}: " + " ".join(render_slot_sentences(missing_a)))
    if missing_b:
        sections.append(f"For {intent_b}: " + " ".join(render_slot_sentences(missing_b)))
    return " ".join(sections) if sections else "Those are the details."


def format_tool_call(
    call_id: str,
    name: str,
    arguments: dict,
    content: str | None = None,
    intent_id: str | None = None,
    phase: str = "execute",
    order: int = 0,
) -> dict:
    tool_calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }
    ]
    task = arguments.get("task", "execute")
    plan_steps = [
        make_step(
            step_id=f"{call_id}_plan",
            tool=name,
            task=task,
            intent_ids=[intent_id] if intent_id else [],
            phase=phase,
            slots_used=sorted(arguments.get("slots", {}).keys()),
            order=order,
        )
    ]
    return assistant_message(
        content or "",
        reasoning=make_reasoning(stage=phase, focus_intents=[intent_id] if intent_id else [], summary="Planning tool execution."),
        tool_calls=tool_calls,
        tool_plan=make_tool_plan(plan_steps),
    )


def tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def show_to_user(call_id: str, message: str, focus_intents: List[str]) -> dict:
    steps = [
        make_step(
            step_id=f"{call_id}_plan",
            tool="show_to_user",
            task="summarize_outcome",
            intent_ids=focus_intents,
            phase="final_summary",
            slots_used=[],
            order=99,
            parallel_group="summary",
        )
    ]
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "show_to_user", "arguments": json.dumps({"message": message})},
            }
        ],
        "reasoning": make_reasoning(
            stage="final_summary",
            focus_intents=focus_intents,
            summary="Share concise summary with the user.",
        ),
        "tool_plan": make_tool_plan(steps),
    }


def conversation_template_single(intent: dict, idx: int, llm: Optional[LLMGenerator], persona: Optional[Dict[str, object]] = None) -> dict:
    slots = choose_slots(intent)
    all_slots = {**slots.present, **slots.missing}
    tool_name = default_tool(intent["intent_id"])
    difficulty = random.choice(DIFFICULTY_BUCKETS)

    user_opening = (
        llm.generate_single(intent["description"], slots.present, slots.missing, difficulty)
        if llm
        else build_user_message(intent["description"], slots.present)
    )
    messages: List[dict] = [
        {"role": "system", "content": apply_persona_to_system_prompt(persona)},
        {"role": "user", "content": user_opening},
        assistant_message(
            content=(
                f"I can help with that {intent['intent_id']}. I need a couple details: "
                + ", ".join(slots.missing.keys())
                if slots.missing
                else "I can help with that."
            ),
            reasoning=make_reasoning(
                stage="clarify_slots",
                focus_intents=[intent["intent_id"]],
                summary="Confirm intent and request missing slots before tool calls.",
            ),
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"collect_{idx}_plan",
                        tool="none",
                        task="collect_slots",
                        intent_ids=[intent["intent_id"]],
                        phase="clarify_slots",
                        slots_used=[],
                        order=0,
                        parallel_group="collect",
                        depends_on=[],
                    ),
                    make_step(
                        step_id=f"diag_{idx}_plan",
                        tool=tool_name,
                        task="diagnose_intent",
                        intent_ids=[intent["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots.keys()) or ["pending"],
                        order=1,
                        parallel_group="diag",
                        depends_on=[f"collect_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"resolve_{idx}_plan",
                        tool=tool_name,
                        task="resolve_intent",
                        intent_ids=[intent["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots.keys()) or ["pending"],
                        order=2,
                        parallel_group="resolve",
                        depends_on=[f"diag_{idx}_plan"],
                    ),
                ]
            ),
        ),
    ]

    if slots.missing:
        messages.append({"role": "user", "content": build_user_followup(slots.missing)})

    diagnose_id = f"diag_{idx}"
    resolve_id = f"resolve_{idx}"
    summary_id = f"show_{idx}"

    diagnose_args = {
        "task": "diagnose_intent",
        "intent_id": intent["intent_id"],
        "slots": all_slots,
    }
    messages.append(
        assistant_message(
            content="Let me check what went wrong with your request.",
            reasoning=make_reasoning(
                stage="diagnosis",
                focus_intents=[intent["intent_id"]],
                summary="Run diagnostic tool call with collected slots.",
            ),
            tool_calls=[
                {
                    "id": diagnose_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(diagnose_args)},
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{diagnose_id}_plan",
                        tool=tool_name,
                        task="diagnose_intent",
                        intent_ids=[intent["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots.keys()),
                        order=1,
                        parallel_group="diag",
                    )
                ]
            ),
        )
    )

    def pick_diag(int_obj: dict) -> str:
        path = int_obj.get("path", [])
        keys = []
        if len(path) >= 3:
            keys.append(".".join(path[1:3]))
        if len(path) >= 2:
            keys.append(path[1])
        if path:
            keys.append(path[0])
        for k in keys:
            if k in INTENT_FAMILY_DIAG:
                return random.choice(INTENT_FAMILY_DIAG[k])
        return "issue detected"

    def pick_res(int_obj: dict) -> str:
        path = int_obj.get("path", [])
        keys = []
        if len(path) >= 3:
            keys.append(".".join(path[1:3]))
        if len(path) >= 2:
            keys.append(path[1])
        if path:
            keys.append(path[0])
        for k in keys:
            if k in INTENT_FAMILY_RESOLUTION:
                return random.choice(INTENT_FAMILY_RESOLUTION[k])
        return "resolution applied"

    issue = pick_diag(intent)
    messages.append(tool_result(diagnose_id, f"Diagnosis: {issue}. Slots={all_slots}"))

    resolve_args = {
        "task": "resolve_intent",
        "intent_id": intent["intent_id"],
        "slots": all_slots,
        "action": "retry_or_escalate",
    }
    messages.append(
        assistant_message(
            content="I’m applying the fix and retrying now.",
            reasoning=make_reasoning(
                stage="resolution",
                focus_intents=[intent["intent_id"]],
                summary="Execute resolution tool after diagnosis.",
            ),
            tool_calls=[
                {
                    "id": resolve_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(resolve_args)},
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{resolve_id}_plan",
                        tool=tool_name,
                        task="resolve_intent",
                        intent_ids=[intent["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots.keys()),
                        order=2,
                        parallel_group="resolve",
                        depends_on=[f"{diagnose_id}_plan"],
                    )
                ]
            ),
        )
    )

    resolution = pick_res(intent)
    messages.append(tool_result(resolve_id, f"Resolution: {resolution}."))

    summary = (
        f"{intent['description'].capitalize()} — {resolution}. "
        "If anything looks off, I can adjust details and rerun."
    )
    messages.append(show_to_user(summary_id, summary, [intent["intent_id"]]))

    return {
        "conversation_id": f"conv_single_{idx:04d}",
        "metadata": {
            "intents": [intent["intent_id"]],
            "decision_category": "single_intent",
            "difficulty": difficulty,
        },
        "messages": messages,
    }


def conversation_template_multi(intent_a: dict, intent_b: dict, idx: int, llm: Optional[LLMGenerator], persona: Optional[Dict[str, object]] = None) -> dict:
    slots_a = choose_slots(intent_a)
    slots_b = choose_slots(intent_b)
    all_slots_a = {**slots_a.present, **slots_a.missing}
    all_slots_b = {**slots_b.present, **slots_b.missing}
    tool_a = default_tool(intent_a["intent_id"])
    tool_b = default_tool(intent_b["intent_id"])
    difficulty = random.choice(DIFFICULTY_BUCKETS)

    user_opening = (
        llm.generate_multi(
            intent_a["description"],
            intent_b["description"],
            slots_a.present,
            slots_b.present,
            slots_a.missing,
            slots_b.missing,
            difficulty,
        )
        if llm
        else build_user_message(
            intent_a["description"],
            slots_a.present,
            multi_hint=f"I also need help with {intent_b['description']}.",
        )
    )
    messages: List[dict] = [
        {"role": "system", "content": apply_persona_to_system_prompt(persona)},
        {
            "role": "user",
            "content": user_opening,
        },
        assistant_message(
            content=(
                f"I’ll handle both: {intent_a['intent_id']} and {intent_b['intent_id']}. "
                "I need a couple details:"
                f" {', '.join(slots_a.missing.keys()) or 'none'} for the first, "
                f"{', '.join(slots_b.missing.keys()) or 'none'} for the second."
            ),
            reasoning=make_reasoning(
                stage="clarify_slots",
                focus_intents=[intent_a["intent_id"], intent_b["intent_id"]],
                summary="Confirm both intents and collect missing slots before tool calls.",
            ),
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"diag_a_{idx}_plan",
                        tool=tool_a,
                        task="diagnose_intent",
                        intent_ids=[intent_a["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_a.keys()) or ["pending"],
                        order=1,
                        parallel_group="diag",
                        depends_on=[f"collect_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"diag_b_{idx}_plan",
                        tool=tool_b,
                        task="diagnose_intent",
                        intent_ids=[intent_b["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_b.keys()) or ["pending"],
                        order=2,
                        parallel_group="diag",
                        depends_on=[f"collect_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"resolve_a_{idx}_plan",
                        tool=tool_a,
                        task="resolve_intent",
                        intent_ids=[intent_a["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_a.keys()) or ["pending"],
                        order=3,
                        parallel_group="resolve",
                        depends_on=[f"diag_a_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"resolve_b_{idx}_plan",
                        tool=tool_b,
                        task="resolve_intent",
                        intent_ids=[intent_b["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_b.keys()) or ["pending"],
                        order=4,
                        parallel_group="resolve",
                        depends_on=[f"diag_b_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"collect_{idx}_plan",
                        tool="none",
                        task="collect_slots",
                        intent_ids=[intent_a["intent_id"], intent_b["intent_id"]],
                        phase="clarify_slots",
                        slots_used=[],
                        order=0,
                        parallel_group="collect",
                        depends_on=[],
                    ),
                ]
            ),
        ),
    ]

    if slots_a.missing or slots_b.missing:
        messages.append(
            {"role": "user", "content": build_user_followup_multi(intent_a["intent_id"], slots_a.missing, intent_b["intent_id"], slots_b.missing)}
        )

    diag_a_id = f"diag_a_{idx}"
    diag_b_id = f"diag_b_{idx}"
    resolve_a_id = f"resolve_a_{idx}"
    resolve_b_id = f"resolve_b_{idx}"
    summary_id = f"show_multi_{idx}"

    messages.append(
        assistant_message(
            content="Checking the first issue now.",
            reasoning=make_reasoning(
                stage="diagnosis",
                focus_intents=[intent_a["intent_id"]],
                summary="Run diagnosis for first intent.",
            ),
            tool_calls=[
                {
                    "id": diag_a_id,
                    "type": "function",
                    "function": {
                        "name": tool_a,
                        "arguments": json.dumps({"task": "diagnose_intent", "intent_id": intent_a["intent_id"], "slots": all_slots_a}),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{diag_a_id}_plan",
                        tool=tool_a,
                        task="diagnose_intent",
                        intent_ids=[intent_a["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_a.keys()),
                        order=1,
                        parallel_group="diag",
                    )
                ]
            ),
        )
    )
    messages.append(
        assistant_message(
            content="Looking into the second request too.",
            reasoning=make_reasoning(
                stage="diagnosis",
                focus_intents=[intent_b["intent_id"]],
                summary="Run diagnosis for second intent.",
            ),
            tool_calls=[
                {
                    "id": diag_b_id,
                    "type": "function",
                    "function": {
                        "name": tool_b,
                        "arguments": json.dumps({"task": "diagnose_intent", "intent_id": intent_b["intent_id"], "slots": all_slots_b}),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{diag_b_id}_plan",
                        tool=tool_b,
                        task="diagnose_intent",
                        intent_ids=[intent_b["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_b.keys()),
                        order=2,
                        parallel_group="diag",
                    )
                ]
            ),
        )
    )

    issue_a = pick_diag(intent_a)
    issue_b = pick_diag(intent_b)
    messages.append(tool_result(diag_a_id, f"Diagnosis: {issue_a}. Slots={all_slots_a}"))
    messages.append(tool_result(diag_b_id, f"Diagnosis: {issue_b}. Slots={all_slots_b}"))

    messages.append(
        assistant_message(
            content="Applying fix for the first issue.",
            reasoning=make_reasoning(
                stage="resolution",
                focus_intents=[intent_a["intent_id"]],
                summary="Execute resolution for first intent.",
            ),
            tool_calls=[
                {
                    "id": resolve_a_id,
                    "type": "function",
                    "function": {
                        "name": tool_a,
                        "arguments": json.dumps(
                            {"task": "resolve_intent", "intent_id": intent_a["intent_id"], "slots": all_slots_a, "action": "resolve"}
                        ),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{resolve_a_id}_plan",
                        tool=tool_a,
                        task="resolve_intent",
                        intent_ids=[intent_a["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_a.keys()),
                        order=3,
                        parallel_group="resolve",
                        depends_on=[f"{diag_a_id}_plan"],
                    )
                ]
            ),
        )
    )
    messages.append(
        assistant_message(
            content="Applying fix for the second request.",
            reasoning=make_reasoning(
                stage="resolution",
                focus_intents=[intent_b["intent_id"]],
                summary="Execute resolution for second intent.",
            ),
            tool_calls=[
                {
                    "id": resolve_b_id,
                    "type": "function",
                    "function": {
                        "name": tool_b,
                        "arguments": json.dumps(
                            {"task": "resolve_intent", "intent_id": intent_b["intent_id"], "slots": all_slots_b, "action": "resolve"}
                        ),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{resolve_b_id}_plan",
                        tool=tool_b,
                        task="resolve_intent",
                        intent_ids=[intent_b["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_b.keys()),
                        order=4,
                        parallel_group="resolve",
                        depends_on=[f"{diag_b_id}_plan"],
                    )
                ]
            ),
        )
    )

    resolution_a = pick_res(intent_a)
    resolution_b = pick_res(intent_b)
    messages.append(tool_result(resolve_a_id, f"Resolution: {resolution_a}."))
    messages.append(tool_result(resolve_b_id, f"Resolution: {resolution_b}."))

    summary = (
        f"{intent_a['intent_id']}: {resolution_a}. "
        f"{intent_b['intent_id']}: {resolution_b}. "
        "Tell me if anything needs another pass."
    )
    messages.append(show_to_user(summary_id, summary, [intent_a["intent_id"], intent_b["intent_id"]]))

    return {
        "conversation_id": f"conv_multi_{idx:04d}",
        "metadata": {
            "intents": [intent_a["intent_id"], intent_b["intent_id"]],
            "decision_category": "multi_intent",
            "difficulty": difficulty,
        },
        "messages": messages,
    }


def conversation_template_tri(intent_a: dict, intent_b: dict, intent_c: dict, idx: int, llm: Optional[LLMGenerator], persona: Optional[Dict[str, object]] = None) -> dict:
    slots_a = choose_slots(intent_a)
    slots_b = choose_slots(intent_b)
    slots_c = choose_slots(intent_c)
    all_slots_a = {**slots_a.present, **slots_a.missing}
    all_slots_b = {**slots_b.present, **slots_b.missing}
    all_slots_c = {**slots_c.present, **slots_c.missing}
    tool_a = default_tool(intent_a["intent_id"])
    tool_b = default_tool(intent_b["intent_id"])
    tool_c = default_tool(intent_c["intent_id"])
    difficulty = random.choice(DIFFICULTY_BUCKETS)

    user_opening = (
        llm.generate_multi(
            intent_a["description"],
            intent_b["description"] + " and " + intent_c["description"],
            slots_a.present,
            {**slots_b.present, **slots_c.present},
            slots_a.missing,
            {**slots_b.missing, **slots_c.missing},
            difficulty,
        )
        if llm
        else build_user_message(
            intent_a["description"],
            slots_a.present,
            multi_hint=f"I also need help with {intent_b['description']} and {intent_c['description']}.",
        )
    )
    messages: List[dict] = [
        {"role": "system", "content": apply_persona_to_system_prompt(persona)},
        {"role": "user", "content": user_opening},
        assistant_message(
            content=(
                f"I’ll handle all three: {intent_a['intent_id']}, {intent_b['intent_id']}, and {intent_c['intent_id']}. "
                "I need a couple details for each."
            ),
            reasoning=make_reasoning(
                stage="clarify_slots",
                focus_intents=[intent_a["intent_id"], intent_b["intent_id"], intent_c["intent_id"]],
                summary="Confirm all three intents and collect missing slots before tool calls.",
            ),
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"collect_{idx}_plan",
                        tool="none",
                        task="collect_slots",
                        intent_ids=[intent_a["intent_id"], intent_b["intent_id"], intent_c["intent_id"]],
                        phase="clarify_slots",
                        slots_used=[],
                        order=0,
                        parallel_group="collect",
                        depends_on=[],
                    ),
                    make_step(
                        step_id=f"diag_a_{idx}_plan",
                        tool=tool_a,
                        task="diagnose_intent",
                        intent_ids=[intent_a["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_a.keys()) or ["pending"],
                        order=1,
                        parallel_group="diag",
                        depends_on=[f"collect_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"diag_b_{idx}_plan",
                        tool=tool_b,
                        task="diagnose_intent",
                        intent_ids=[intent_b["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_b.keys()) or ["pending"],
                        order=2,
                        parallel_group="diag",
                        depends_on=[f"collect_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"diag_c_{idx}_plan",
                        tool=tool_c,
                        task="diagnose_intent",
                        intent_ids=[intent_c["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_c.keys()) or ["pending"],
                        order=3,
                        parallel_group="diag",
                        depends_on=[f"collect_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"resolve_a_{idx}_plan",
                        tool=tool_a,
                        task="resolve_intent",
                        intent_ids=[intent_a["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_a.keys()) or ["pending"],
                        order=4,
                        parallel_group="resolve",
                        depends_on=[f"diag_a_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"resolve_b_{idx}_plan",
                        tool=tool_b,
                        task="resolve_intent",
                        intent_ids=[intent_b["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_b.keys()) or ["pending"],
                        order=5,
                        parallel_group="resolve",
                        depends_on=[f"diag_b_{idx}_plan"],
                    ),
                    make_step(
                        step_id=f"resolve_c_{idx}_plan",
                        tool=tool_c,
                        task="resolve_intent",
                        intent_ids=[intent_c["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_c.keys()) or ["pending"],
                        order=6,
                        parallel_group="resolve",
                        depends_on=[f"diag_c_{idx}_plan"],
                    ),
                ]
            ),
        ),
    ]

    if slots_a.missing or slots_b.missing or slots_c.missing:
        parts = []
        if slots_a.missing:
            parts.append(f"For {intent_a['intent_id']}: " + " ".join(render_slot_sentences(slots_a.missing)))
        if slots_b.missing:
            parts.append(f"For {intent_b['intent_id']}: " + " ".join(render_slot_sentences(slots_b.missing)))
        if slots_c.missing:
            parts.append(f"For {intent_c['intent_id']}: " + " ".join(render_slot_sentences(slots_c.missing)))
        messages.append({"role": "user", "content": " ".join(parts)})

    diag_a_id = f"diag_a_{idx}"
    diag_b_id = f"diag_b_{idx}"
    diag_c_id = f"diag_c_{idx}"
    resolve_a_id = f"resolve_a_{idx}"
    resolve_b_id = f"resolve_b_{idx}"
    resolve_c_id = f"resolve_c_{idx}"
    summary_id = f"show_tri_{idx}"

    messages.append(
        assistant_message(
            content="Checking the first issue now.",
            reasoning=make_reasoning(
                stage="diagnosis",
                focus_intents=[intent_a["intent_id"]],
                summary="Run diagnosis for first intent.",
            ),
            tool_calls=[
                {
                    "id": diag_a_id,
                    "type": "function",
                    "function": {
                        "name": tool_a,
                        "arguments": json.dumps({"task": "diagnose_intent", "intent_id": intent_a["intent_id"], "slots": all_slots_a}),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{diag_a_id}_plan",
                        tool=tool_a,
                        task="diagnose_intent",
                        intent_ids=[intent_a["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_a.keys()),
                        order=1,
                        parallel_group="diag",
                    )
                ]
            ),
        )
    )
    messages.append(
        assistant_message(
            content="Looking into the second request too.",
            reasoning=make_reasoning(
                stage="diagnosis",
                focus_intents=[intent_b["intent_id"]],
                summary="Run diagnosis for second intent.",
            ),
            tool_calls=[
                {
                    "id": diag_b_id,
                    "type": "function",
                    "function": {
                        "name": tool_b,
                        "arguments": json.dumps({"task": "diagnose_intent", "intent_id": intent_b["intent_id"], "slots": all_slots_b}),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{diag_b_id}_plan",
                        tool=tool_b,
                        task="diagnose_intent",
                        intent_ids=[intent_b["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_b.keys()),
                        order=2,
                        parallel_group="diag",
                    )
                ]
            ),
        )
    )
    messages.append(
        assistant_message(
            content="Reviewing the third request now.",
            reasoning=make_reasoning(
                stage="diagnosis",
                focus_intents=[intent_c["intent_id"]],
                summary="Run diagnosis for third intent.",
            ),
            tool_calls=[
                {
                    "id": diag_c_id,
                    "type": "function",
                    "function": {
                        "name": tool_c,
                        "arguments": json.dumps({"task": "diagnose_intent", "intent_id": intent_c["intent_id"], "slots": all_slots_c}),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{diag_c_id}_plan",
                        tool=tool_c,
                        task="diagnose_intent",
                        intent_ids=[intent_c["intent_id"]],
                        phase="diagnosis",
                        slots_used=sorted(all_slots_c.keys()),
                        order=3,
                        parallel_group="diag",
                    )
                ]
            ),
        )
    )

    issue_a = pick_diag(intent_a)
    issue_b = pick_diag(intent_b)
    issue_c = pick_diag(intent_c)
    messages.append(tool_result(diag_a_id, f"Diagnosis: {issue_a}. Slots={all_slots_a}"))
    messages.append(tool_result(diag_b_id, f"Diagnosis: {issue_b}. Slots={all_slots_b}"))
    messages.append(tool_result(diag_c_id, f"Diagnosis: {issue_c}. Slots={all_slots_c}"))

    messages.append(
        assistant_message(
            content="Applying fix for the first issue.",
            reasoning=make_reasoning(
                stage="resolution",
                focus_intents=[intent_a["intent_id"]],
                summary="Execute resolution for first intent.",
            ),
            tool_calls=[
                {
                    "id": resolve_a_id,
                    "type": "function",
                    "function": {
                        "name": tool_a,
                        "arguments": json.dumps(
                            {"task": "resolve_intent", "intent_id": intent_a["intent_id"], "slots": all_slots_a, "action": "resolve"}
                        ),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{resolve_a_id}_plan",
                        tool=tool_a,
                        task="resolve_intent",
                        intent_ids=[intent_a["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_a.keys()),
                        order=4,
                        parallel_group="resolve",
                        depends_on=[f"{diag_a_id}_plan"],
                    )
                ]
            ),
        )
    )
    messages.append(
        assistant_message(
            content="Applying fix for the second request.",
            reasoning=make_reasoning(
                stage="resolution",
                focus_intents=[intent_b["intent_id"]],
                summary="Execute resolution for second intent.",
            ),
            tool_calls=[
                {
                    "id": resolve_b_id,
                    "type": "function",
                    "function": {
                        "name": tool_b,
                        "arguments": json.dumps(
                            {"task": "resolve_intent", "intent_id": intent_b["intent_id"], "slots": all_slots_b, "action": "resolve"}
                        ),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{resolve_b_id}_plan",
                        tool=tool_b,
                        task="resolve_intent",
                        intent_ids=[intent_b["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_b.keys()),
                        order=5,
                        parallel_group="resolve",
                        depends_on=[f"{diag_b_id}_plan"],
                    )
                ]
            ),
        )
    )
    messages.append(
        assistant_message(
            content="Applying fix for the third issue.",
            reasoning=make_reasoning(
                stage="resolution",
                focus_intents=[intent_c["intent_id"]],
                summary="Execute resolution for third intent.",
            ),
            tool_calls=[
                {
                    "id": resolve_c_id,
                    "type": "function",
                    "function": {
                        "name": tool_c,
                        "arguments": json.dumps(
                            {"task": "resolve_intent", "intent_id": intent_c["intent_id"], "slots": all_slots_c, "action": "resolve"}
                        ),
                    },
                }
            ],
            tool_plan=make_tool_plan(
                [
                    make_step(
                        step_id=f"{resolve_c_id}_plan",
                        tool=tool_c,
                        task="resolve_intent",
                        intent_ids=[intent_c["intent_id"]],
                        phase="resolution",
                        slots_used=sorted(all_slots_c.keys()),
                        order=6,
                        parallel_group="resolve",
                        depends_on=[f"{diag_c_id}_plan"],
                    )
                ]
            ),
        )
    )

    resolution_a = pick_res(intent_a)
    resolution_b = pick_res(intent_b)
    resolution_c = pick_res(intent_c)
    messages.append(tool_result(resolve_a_id, f"Resolution: {resolution_a}."))
    messages.append(tool_result(resolve_b_id, f"Resolution: {resolution_b}."))
    messages.append(tool_result(resolve_c_id, f"Resolution: {resolution_c}."))

    summary = (
        f"{intent_a['intent_id']}: {resolution_a}. "
        f"{intent_b['intent_id']}: {resolution_b}. "
        f"{intent_c['intent_id']}: {resolution_c}. "
        "Tell me if anything needs another pass."
    )
    messages.append(show_to_user(summary_id, summary, [intent_a["intent_id"], intent_b["intent_id"], intent_c["intent_id"]]))

    return {
        "conversation_id": f"conv_tri_{idx:04d}",
        "metadata": {
            "intents": [intent_a["intent_id"], intent_b["intent_id"], intent_c["intent_id"]],
            "decision_category": "multi_intent",
            "difficulty": difficulty,
        },
        "messages": messages,
    }


def generate_conversations(count: int, multi_ratio: float, llm: Optional[LLMGenerator], persona: Optional[Dict[str, object]]) -> List[dict]:
    conversations: List[dict] = []
    for idx in range(count):
        is_multi = random.random() < multi_ratio
        if is_multi and len(SYNTHETIC_INTENTS) >= 2:
            a, b = choose_intent_pair()
            conversations.append(conversation_template_multi(a, b, idx, llm, persona))
        else:
            intent = random.choice(SYNTHETIC_INTENTS)
            conversations.append(conversation_template_single(intent, idx, llm, persona))
    return conversations


def save_jsonl(rows: Sequence[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-turn conversations for tool planning.")
    parser.add_argument("--output", type=Path, default=Path("examples/mock_conversation_traces.jsonl"), help="Path to write JSONL conversations.")
    parser.add_argument("--count", type=int, default=5, help="Number of conversations to generate.")
    parser.add_argument("--multi-ratio", type=float, default=0.35, help="Fraction of conversations that are multi-intent.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--use-llm", action="store_true", help="Use an LLM (liteLLM) to generate user utterances.")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model name (OpenAI-compatible via liteLLM).")
    parser.add_argument("--llm-base-url", type=str, default=None, help="Optional custom base URL for OpenAI-compatible endpoints.")
    parser.add_argument("--llm-api-key-env", type=str, default="LITELLM_API_KEY", help="Env var holding the primary API key (falls back to OPENAI_API_KEY).")
    parser.add_argument("--llm-temperature", type=float, default=0.8, help="LLM temperature.")
    parser.add_argument("--llm-max-tokens", type=int, default=80, help="Max tokens for LLM completion.")
    parser.add_argument("--no-llm-drop-params", action="store_false", dest="llm_drop_params", help="Keep unsupported params instead of dropping them.")
    parser.add_argument("--persona", type=str, default="concise_support", help="Persona id to apply (see PERSONAS).")
    parser.add_argument("--persona-random", action="store_true", help="Pick a random persona from PERSONAS.")
    args = parser.parse_args()

    random.seed(args.seed)
    llm: Optional[LLMGenerator] = None
    if args.use_llm:
        try:
            llm = LLMGenerator(
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
                drop_params=args.llm_drop_params,
            )
            print(f"LLM generation enabled: model={args.llm_model} base_url={args.llm_base_url or 'default'}")
            if "gpt-5" in args.llm_model and args.llm_temperature != 1:
                print("Note: temperature reset to 1 for gpt-5 models to match API constraints.")
        except Exception as exc:
            raise SystemExit(f"Failed to initialize LLM generator: {exc}") from exc

    persona: Optional[Dict[str, object]] = None
    if args.persona_random:
        persona = random.choice(list(PERSONAS.values()))
    else:
        persona = PERSONAS.get(args.persona, PERSONAS.get("concise_support"))

    conversations = generate_conversations(args.count, args.multi_ratio, llm, persona)
    save_jsonl(conversations, args.output)
    print(f"Wrote {len(conversations)} conversations to {args.output}")


if __name__ == "__main__":
    main()
