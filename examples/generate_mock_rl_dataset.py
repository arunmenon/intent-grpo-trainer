"""
Generate a synthetic RL dataset (DecisionObject schema) using the mock KG.

Curriculum targets (scaled to requested count, default 500):
- Single intent, simple/pick: 150
- Single intent, medium/clarify: 140
- Single intent, complex/clarify: 90
- Single intent, vague/clarify: 50
- Multi intent, medium/multi_intent: 50
- Multi intent, complex/multi_intent: 20

Fields per row:
- prompt (string)
- gold_intents (list[str])
- known_paths (list[list[str]])
- gold_decision ("pick" | "clarify" | "multi_intent")
- ambiguous_slots (list[str])
- clarification_targets (list[str])
- reasoning_terms (list[str])
- candidate_intents (list[str])
- difficulty (simple|medium|complex|vague)
- decision_category (single_intent|multi_intent)
- source ("synthetic")

Usage:
  python3 examples/generate_mock_rl_dataset.py --output examples/mock_rl_dataset.jsonl --count 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from examples.synthetic_kg import SYNTHETIC_INTENTS as KG_INTENTS


RANDOM_SEED = 13
BASE_TOTAL = 500

SLOT_VALUE_POOL: Dict[str, List[str]] = {
    "txn_id": ["9AB123", "PAY-7788", "TXN-4420", "INV-9931", "TRX-3322"],
    "amount": ["20 dollars", "75 USD", "150 bucks", "42.50 USD", "300 dollars"],
    "currency": ["USD", "EUR", "GBP"],
    "reason": ["duplicate charge", "wrong item", "buyer requested", "shipping delay", "fraud concern"],
    "date": ["yesterday", "two days ago", "last week", "Oct 10", "Nov 2"],
    "receiver": ["my friend Alex", "john@example.com", "Mom", "a vendor", "my roommate"],
    "device": ["mobile app", "desktop browser", "tablet", "work laptop"],
    "location": ["California", "London", "Berlin", "Toronto"],
    "merchant": ["Etsy seller", "Uber", "eBay", "Airbnb host"],
    "seller": ["an Etsy shop", "a Shopify store", "eBay seller", "marketplace vendor"],
    "expected_delivery_date": ["tomorrow", "next Monday", "Oct 12", "Nov 3"],
    "shipping_carrier": ["UPS", "FedEx", "USPS", "DHL"],
    "tracking_id": ["1Z999", "9400-ABCD", "TRK-1122"],
    "photos": ["photo attached", "can upload photos", "screenshots ready"],
    "issue_detail": ["wrong color", "missing parts", "damaged on arrival", "different model"],
    "resolution_preference": ["refund", "replacement"],
    "case_id": ["CB-1200", "CB-8891", "CB-4433"],
    "evidence_type": ["proof of delivery", "proof of refund", "communication logs"],
    "deadline_date": ["in 3 days", "Oct 20", "Nov 5"],
    "card_last4": ["1234", "9876", "0011", "5599"],
    "issuer": ["Chase", "Citi", "Amex", "Wells Fargo"],
    "error_message": ["card not supported", "verification failed", "bank declined"],
    "bank_name": ["Chase", "Bank of America", "HSBC"],
    "country": ["US", "UK", "Canada"],
    "login_email": ["user@example.com", "buyer@paypal.com", "seller@shop.com"],
    "recovery_method": ["email recovery", "sms recovery"],
    "next_bill_date": ["next week", "Oct 15", "Nov 1"],
}

NOISE_SNIPPETS = [
    "I'm kind of stressed about this.",
    "Please keep it short.",
    "This is time-sensitive.",
    "I already checked my bank.",
    "I don't want to open a dispute if I can avoid it.",
    "Can we fix this quickly?",
    "I tried support chat but it timed out.",
]

INTENT_INDEX = {i["intent_id"]: i for i in KG_INTENTS}

# Prioritize plausible intent pairs; allow small fraction of noise pairs for robustness.
PLAUSIBLE_PAIR_IDS: List[Tuple[str, str]] = [
    ("PPA.Disputes.Unauthorized", "PPA.Account.Access.LockedOut"),
    ("PPA.Disputes.Unauthorized", "PPA.Account.Security.TwoFactorReset"),
    ("PPA.Disputes.Unauthorized", "PPA.Refunds.BuyerRefunds.PartialRefund"),
    ("PPA.Disputes.ItemIssues.ItemNotReceived", "PPA.Refunds.BuyerRefunds.PartialRefund"),
    ("PPA.Disputes.ItemIssues.NotAsDescribed", "PPA.Refunds.BuyerRefunds.FullRefund"),
    ("PPA.Disputes.ItemIssues.ItemNotReceived", "PPA.Disputes.ItemIssues.NotAsDescribed"),
    ("PPA.Payments.SendMoney.Issue", "PPA.Payments.BankLink.Verification"),
    ("PPA.Payments.AddCard.Failed", "PPA.Payments.BankLink.Verification"),
    ("PPA.Account.Access.LockedOut", "PPA.Account.Security.TwoFactorReset"),
    ("PPA.Payments.Subscriptions.CancelRecurring", "PPA.Refunds.BuyerRefunds.PartialRefund"),
    ("PPA.Payments.Subscriptions.CancelRecurring", "PPA.Refunds.BuyerRefunds.FullRefund"),
]

# Noise pool: any remaining unique pairs not in plausible list.
INTENT_IDS = [i["intent_id"] for i in KG_INTENTS]
ALL_PAIRS = [(a, b) for a, b in combinations(INTENT_IDS, 2)]
PLAUSIBLE_SET = {tuple(sorted(p)) for p in PLAUSIBLE_PAIR_IDS}
NOISE_PAIR_IDS = [p for p in ALL_PAIRS if tuple(sorted(p)) not in PLAUSIBLE_SET]


@dataclass
class Bucket:
    decision_category: str
    difficulty: str
    gold_decision: str
    missing_slots: int
    noise: int
    target: int


BUCKETS: List[Bucket] = [
    Bucket("single_intent", "simple", "pick", missing_slots=0, noise=0, target=150),
    Bucket("single_intent", "medium", "clarify", missing_slots=1, noise=0, target=140),
    Bucket("single_intent", "complex", "clarify", missing_slots=2, noise=1, target=90),
    Bucket("single_intent", "vague", "clarify", missing_slots=3, noise=2, target=50),
    Bucket("multi_intent", "medium", "multi_intent", missing_slots=2, noise=1, target=50),
    Bucket("multi_intent", "complex", "multi_intent", missing_slots=3, noise=2, target=20),
]


def pick_slot_value(slot: str) -> str:
    pool = SLOT_VALUE_POOL.get(slot, [slot])
    return random.choice(pool)


def build_slot_sentences(slots: Iterable[str]) -> List[str]:
    sentences = []
    for slot in slots:
        value = pick_slot_value(slot)
        if slot == "amount":
            sentences.append(f"I want to move {value}.")
        elif slot == "currency":
            sentences.append(f"It is in {value}.")
        elif slot == "txn_id":
            sentences.append(f"The transaction ID is {value}.")
        elif slot == "receiver":
            sentences.append(f"It was to {value}.")
        elif slot == "reason":
            sentences.append(f"The reason is {value}.")
        elif slot == "date":
            sentences.append(f"It happened {value}.")
        elif slot == "device":
            sentences.append(f"I was on the {value}.")
        elif slot == "location":
            sentences.append(f"I was in {value}.")
        elif slot == "merchant":
            sentences.append(f"The merchant was {value}.")
        elif slot == "seller":
            sentences.append(f"The seller was {value}.")
        elif slot == "expected_delivery_date":
            sentences.append(f"It was due {value}.")
        elif slot == "shipping_carrier":
            sentences.append(f"It shipped via {value}.")
        elif slot == "tracking_id":
            sentences.append(f"The tracking number is {value}.")
        elif slot == "photos":
            sentences.append(f"I have {value}.")
        elif slot == "issue_detail":
            sentences.append(f"The issue is {value}.")
        elif slot == "resolution_preference":
            sentences.append(f"I prefer a {value}.")
        elif slot == "case_id":
            sentences.append(f"The chargeback case ID is {value}.")
        elif slot == "evidence_type":
            sentences.append(f"I can provide {value}.")
        elif slot == "deadline_date":
            sentences.append(f"The deadline is {value}.")
        elif slot == "card_last4":
            sentences.append(f"The card ends with {value}.")
        elif slot == "issuer":
            sentences.append(f"The issuer is {value}.")
        elif slot == "error_message":
            sentences.append(f"I saw the error '{value}'.")
        elif slot == "bank_name":
            sentences.append(f"The bank is {value}.")
        elif slot == "country":
            sentences.append(f"The account country is {value}.")
        elif slot == "login_email":
            sentences.append(f"My login email is {value}.")
        elif slot == "recovery_method":
            sentences.append(f"I can use {value}.")
        elif slot == "next_bill_date":
            sentences.append(f"The next billing date is {value}.")
        else:
            sentences.append(f"{slot}: {value}")
    return sentences


def synthesize_prompt(intent_desc: str, present_slots: Sequence[str], difficulty: str) -> str:
    slot_sentences = build_slot_sentences(present_slots)
    if difficulty == "vague":
        opener = f"I think this is about {intent_desc}, but I'm not sure."
    else:
        opener = f"I need help with {intent_desc}."
    text = " ".join([opener] + slot_sentences)
    return text


def derive_clarifications(intent: dict, missing_slots: Iterable[str]) -> List[str]:
    templates = intent.get("clarification_templates", {})
    questions: List[str] = []
    for slot in missing_slots:
        tmpl = templates.get(slot, f"What is the {slot}?")
        questions.append(tmpl)
    return questions


def reasoning_terms(intent: dict, missing_slots: Iterable[str]) -> List[str]:
    terms = [word for word in intent["description"].split() if len(word) > 3][:3]
    terms.extend(missing_slots)
    return list(dict.fromkeys([t.lower() for t in terms]))


def candidate_intents(intent_id: str) -> List[str]:
    others = [i["intent_id"] for i in KG_INTENTS if i["intent_id"] != intent_id]
    return others[:4]


def dedup(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class LLMGenerator:
    """
    Optional LLM-backed utterance generator. Requires openai-compatible SDK and API key.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.8,
        max_tokens: int = 80,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("openai package is required for LLM generation") from exc

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"API key not found in env var {api_key_env}")

        self.client = OpenAI(api_key=api_key, base_url=base_url)  # base_url None -> default
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, intent_desc: str, present_slots: Sequence[str], missing_slots: Sequence[str], difficulty: str) -> str:
        sys_prompt = (
            "You generate concise, single-turn user utterances for PPA intent routing. "
            "Keep it natural, 1-2 sentences, no JSON. Include provided slot values. "
            "If slots are missing, the utterance should naturally omit them. "
            "Match the requested difficulty: simple (direct), medium (one slot missing), "
            "complex (longer, multiple details), vague (underspecified)."
        )
        slot_text = "; ".join(build_slot_sentences(present_slots)) if present_slots else "no slot values"
        miss_text = ", ".join(missing_slots) if missing_slots else "none"
        user_prompt = (
            f"Intent: {intent_desc}\n"
            f"Difficulty: {difficulty}\n"
            f"Given slot values to include: {slot_text}\n"
            f"Missing/ambiguous slots: {miss_text}\n"
            "Return only the user utterance."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()


def choose_intent_pair(noise_ratio: float) -> tuple[dict, dict, str]:
    """
    Select a pair of intents for multi-intent examples, preferring plausible pairs.
    Returns (intent_a, intent_b, pair_type).
    """
    use_noise = random.random() < noise_ratio and NOISE_PAIR_IDS
    pair_ids = random.choice(NOISE_PAIR_IDS if use_noise else PLAUSIBLE_PAIR_IDS)
    pair_type = "noise" if use_noise else "plausible"
    return INTENT_INDEX[pair_ids[0]], INTENT_INDEX[pair_ids[1]], pair_type


def build_single_example(intent: dict, bucket: Bucket, counter: int, llm: Optional[LLMGenerator]) -> dict:
    required = intent["required_slots"]
    missing_count = min(bucket.missing_slots, len(required))
    missing = random.sample(required, missing_count) if missing_count else []
    present = [s for s in required if s not in missing]
    if llm:
        prompt = llm.generate(intent["description"], present_slots=present, missing_slots=missing, difficulty=bucket.difficulty)
    else:
        prompt = synthesize_prompt(intent["description"], present_slots=present, difficulty=bucket.difficulty)
    if bucket.noise:
        noise = random.sample(NOISE_SNIPPETS, k=min(bucket.noise, len(NOISE_SNIPPETS)))
        prompt = f"{prompt} {' '.join(noise)}"
    clarifications = derive_clarifications(intent, missing)
    example_id = f"single__{intent['intent_id'].replace('.', '_').lower()}__{bucket.difficulty}__{counter}"
    return {
        "example_id": example_id,
        "prompt": prompt,
        "gold_intents": [intent["intent_id"]],
        "known_paths": [intent["path"]],
        "gold_decision": bucket.gold_decision,
        "ambiguous_slots": missing,
        "clarification_targets": clarifications,
        "reasoning_terms": reasoning_terms(intent, missing),
        "candidate_intents": candidate_intents(intent["intent_id"]),
        "difficulty": bucket.difficulty,
        "decision_category": "single_intent",
        "source": "synthetic",
    }


def build_multi_example(intent_a: dict, intent_b: dict, bucket: Bucket, counter: int, pair_type: str, llm: Optional[LLMGenerator]) -> dict:
    combined_desc = f"{intent_a['description']} and {intent_b['description']}"
    required = intent_a["required_slots"] + intent_b["required_slots"]
    missing_count = min(bucket.missing_slots, len(required))
    missing = random.sample(required, missing_count) if missing_count else []
    present = [s for s in required if s not in missing][:5]
    if llm:
        prompt_core = llm.generate(combined_desc, present_slots=present, missing_slots=missing, difficulty=bucket.difficulty)
    else:
        prompt_core = synthesize_prompt(combined_desc, present_slots=present, difficulty=bucket.difficulty)
    prompt = f"I need help with two things: {prompt_core}"
    if bucket.noise:
        noise = random.sample(NOISE_SNIPPETS, k=min(bucket.noise, len(NOISE_SNIPPETS)))
        prompt = f"{prompt} {' '.join(noise)}"
    missing_a = [s for s in missing if s in intent_a["required_slots"]]
    missing_b = [s for s in missing if s in intent_b["required_slots"]]
    clarifications = dedup(derive_clarifications(intent_a, missing_a) + derive_clarifications(intent_b, missing_b))
    reasoning = dedup(reasoning_terms(intent_a, missing_a) + reasoning_terms(intent_b, missing_b))
    example_id = f"multi__{bucket.difficulty}__{counter}"
    return {
        "example_id": example_id,
        "prompt": prompt,
        "gold_intents": [intent_a["intent_id"], intent_b["intent_id"]],
        "known_paths": [intent_a["path"], intent_b["path"]],
        "gold_decision": "multi_intent",
        "ambiguous_slots": dedup(missing),
        "clarification_targets": clarifications,
        "reasoning_terms": reasoning,
        "candidate_intents": dedup(candidate_intents(intent_a["intent_id"]) + candidate_intents(intent_b["intent_id"])),
        "difficulty": bucket.difficulty,
        "decision_category": "multi_intent",
        "source": "synthetic",
        "pair_type": pair_type,
    }


def scaled_targets(total: int) -> List[Bucket]:
    scale = total / BASE_TOTAL
    scaled = []
    for b in BUCKETS:
        scaled_count = max(1, round(b.target * scale))
        scaled.append(Bucket(b.decision_category, b.difficulty, b.gold_decision, b.missing_slots, b.noise, scaled_count))
    # Adjust if rounding drifted.
    delta = total - sum(b.target for b in scaled)
    if delta != 0:
        # Nudge the largest bucket(s) while keeping counts >=1.
        sorted_idx = sorted(range(len(scaled)), key=lambda i: scaled[i].target, reverse=True)
        idx = 0
        while delta != 0 and idx < len(sorted_idx):
            target = scaled[sorted_idx[idx]].target
            if delta > 0:
                scaled[sorted_idx[idx]].target = target + 1
                delta -= 1
            else:
                if target > 1:
                    scaled[sorted_idx[idx]].target = target - 1
                    delta += 1
            idx = (idx + 1) % len(sorted_idx)
    return scaled


def build_dataset(target_count: int, llm: Optional[LLMGenerator]) -> List[dict]:
    buckets = scaled_targets(target_count)
    dataset: List[dict] = []
    counter = 0
    noise_ratio = 0.15  # fraction of multi-intent pairs drawn from noise pool
    for bucket in buckets:
        for _ in range(bucket.target):
            counter += 1
            if bucket.decision_category == "single_intent":
                intent = random.choice(KG_INTENTS)
                dataset.append(build_single_example(intent, bucket, counter, llm))
            else:
                intent_a, intent_b, pair_type = choose_intent_pair(noise_ratio)
                dataset.append(build_multi_example(intent_a, intent_b, bucket, counter, pair_type, llm))
    random.shuffle(dataset)
    return dataset[:target_count]


def save_jsonl(examples: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(dataset: List[dict]) -> str:
    decisions = {}
    difficulty = {}
    category = {}
    pair_types = {"plausible": 0, "noise": 0}
    for row in dataset:
        decisions[row["gold_decision"]] = decisions.get(row["gold_decision"], 0) + 1
        difficulty[row["difficulty"]] = difficulty.get(row["difficulty"], 0) + 1
        category[row["decision_category"]] = category.get(row["decision_category"], 0) + 1
        if row["decision_category"] == "multi_intent":
            pair_types[row.get("pair_type", "plausible")] += 1
    return f"Decisions: {decisions} | Difficulty: {difficulty} | Categories: {category} | Multi pair types: {pair_types}"


def main():
    parser = argparse.ArgumentParser(description="Generate mock RL dataset for DecisionObject reward.")
    parser.add_argument("--output", type=Path, default=Path("examples/mock_rl_dataset.jsonl"), help="JSONL output path.")
    parser.add_argument("--count", type=int, default=500, help="Number of examples to emit (default 500).")
    parser.add_argument("--use-llm", action="store_true", help="Use an LLM to generate utterances instead of templates.")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model name (OpenAI-compatible).")
    parser.add_argument("--llm-base-url", type=str, default=None, help="Optional custom base URL for OpenAI-compatible endpoints.")
    parser.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY", help="Env var holding the API key.")
    parser.add_argument("--llm-temperature", type=float, default=0.8, help="LLM temperature.")
    parser.add_argument("--llm-max-tokens", type=int, default=80, help="Max tokens for LLM completion.")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
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
        except Exception as exc:
            raise SystemExit(f"Failed to initialize LLM generator: {exc}") from exc

    dataset = build_dataset(args.count, llm)
    save_jsonl(dataset, args.output)

    print(f"Wrote {len(dataset)} examples to {args.output}")
    print(summarize(dataset))


if __name__ == "__main__":
    main()
