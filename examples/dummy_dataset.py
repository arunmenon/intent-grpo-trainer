"""
Creates a tiny dummy dataset matching the TRL GRPO reward wrappers.

Fields per example:
  - prompt: user query text
  - gold_intents: list[str] (expected KG path/id(s))
  - gold_decision: "pick" | "clarify" | "multi_intent"
  - ambiguous_slots: list[str] (keywords the clarifying question should cover)
  - reasoning_terms: list[str] (keywords expected in reasoning)
  - known_paths: list[str] (valid intents for KG check; optional)
"""

from datasets import Dataset


def build_dummy_dataset() -> Dataset:
    data = [
        {
            "prompt": "Looking for men's running shoes size 10",
            "gold_intents": ["shoes/running/mens"],
            "gold_decision": "pick",
            "ambiguous_slots": [],
            "reasoning_terms": ["running shoes", "men", "size 10"],
            "known_paths": ["shoes/running/mens", "shoes/running/womens", "shoes/basketball/mens"],
        },
        {
            "prompt": "Need shoes",
            "gold_intents": [],
            "gold_decision": "clarify",
            "ambiguous_slots": ["type", "gender", "size"],
            "reasoning_terms": ["ambiguous", "type", "gender", "size"],
            "known_paths": ["shoes/running/mens", "shoes/running/womens", "shoes/boots"],
        },
        {
            "prompt": "Lipstick or mascara recommendations",
            "gold_intents": ["beauty/lipstick", "beauty/mascara"],
            "gold_decision": "multi_intent",
            "ambiguous_slots": [],
            "reasoning_terms": ["lipstick", "mascara", "two intents"],
            "known_paths": ["beauty/lipstick", "beauty/mascara", "beauty/foundation"],
        },
    ]
    return Dataset.from_list(data)


if __name__ == "__main__":
    ds = build_dummy_dataset()
    print(ds)
    # Example: save to disk or use directly with GRPOTrainer
    # ds.save_to_disk("dummy_intent_ds")
