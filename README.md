# intent-grpo-trainer

GRPO/TRL scaffold for structured intent disambiguation with JSON schema rewards.

What’s here
- Reward rubric (R1–R5) for JSON validity/KG paths, decision type, intent match, question coverage, and reasoning keywords.
- Schema parsing helpers to safely score partially valid outputs.
- Optional replay cache and rollout scheduler (for custom loops).
- TRL GRPO integration with wrappers and a dummy dataset + smoke test.

Quick start (TRL)
1. Install deps: `pip install -U transformers trl datasets`
2. Build the dummy dataset and run a smoke test:
   ```
   python examples/run_trl_dummy.py
   ```
3. Swap `model_name` and replace the dummy dataset with your own that includes:
   - `prompt`
   - `gold_intents`
   - `gold_decision` (`pick`/`clarify`/`multi_intent`)
   - `ambiguous_slots`
   - `reasoning_terms`
   - optional `known_paths` (KG whitelist)
4. Call `trainer.train()` from `examples/trl_integration.py`, tuning `GRPOConfig` (num_generations, max_completion_length, batch size, bf16/fp16) for your hardware.

Custom loop option
If you need cache mixing or custom scheduling, `grpo_intent/trainer_scaffold.py` and `examples/hook_template.py` show how to plug in your own generate/update hooks.
