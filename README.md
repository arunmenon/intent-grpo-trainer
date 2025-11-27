# intent-grpo-trainer

GRPO/TRL scaffold for structured intent disambiguation with JSON schema rewards. Ships with two reward tracks:
- Legacy `final_intents` rewards (R1–R5).
- DecisionObject rewards (JSON decision_type + picked intents + clarifications + reasoning), aligned to PPA-style intent routing.

What’s here
- Reward rubric (R1–R5) for JSON validity/KG paths, decision type, intent match, question coverage, and reasoning keywords. See `grpo_intent/reward_components.py` (legacy) and `grpo_intent/intent_reward.py` (DecisionObject).
- Schema parsing helpers: `grpo_intent/schema.py` (legacy) and `grpo_intent/decision_schema.py` (DecisionObject) to safely score partially valid outputs.
- Optional replay cache and rollout scheduler (`grpo_intent/replay_cache.py`, `grpo_intent/schedule.py`) plus trainer scaffold (`grpo_intent/trainer_scaffold.py`).
- Dataset loader/validator: `grpo_intent/dataset.py` loads DecisionObject JSONL; `examples/validate_dataset.py` validates against the mock KG.
- TRL GRPO integration with wrappers, dummy dataset, and DecisionObject dataset generator/loader under `examples/`.

Quick start (TRL)
1) Install deps: `pip install -U transformers trl datasets`
2) Legacy smoke test (final_intents schema):
   ```bash
   python examples/run_trl_dummy.py
   ```
3) DecisionObject path (PPA-style):
   - Generate or use the provided dataset: `python3 examples/generate_mock_rl_dataset.py --output examples/mock_rl_dataset.jsonl --count 500`
   - Validate against the KG: `python3 examples/validate_dataset.py --data examples/mock_rl_dataset.jsonl --kg examples/synthetic_kg.json`
   - Optional: synthesize utterances with an LLM via liteLLM by adding `--use-llm --llm-model <model>` (OpenAI-compatible; set API key in `LITELLM_API_KEY` or `OPENAI_API_KEY`, custom base via `--llm-base-url`).
   - Optional: progress logging with `--log-every <N>`, parallel LLM generation with `--llm-workers <num_threads>`, reproducibility via `--seed`, and prompt deduplication via `--dedup-prompts`.
   - Load and train:
     ```python
     from examples.trl_integration import build_trl_grpo_trainer, load_decision_dataset
     ds = load_decision_dataset("examples/mock_rl_dataset.jsonl")
     trainer = build_trl_grpo_trainer("gpt2", ds, reward_variant="decision_object")
     trainer.train(max_steps=1)
     ```
4) Tune `GRPOConfig` (num_generations, max_completion_length, batch size, bf16/fp16) for your hardware in `examples/trl_integration.py`.

Custom loop option
If you need cache mixing or custom scheduling, `grpo_intent/trainer_scaffold.py` and `examples/hook_template.py` show how to plug in your own generate/update hooks.

Data + KG assets (DecisionObject path)
- KG spec: `examples/synthetic_kg.json` (12 PPA-relevant intents with paths, required/optional slots, clarification templates).
- Dataset generator: `examples/generate_mock_rl_dataset.py` (curriculum buckets, plausible multi-intent pairs, optional LLM utterances, small noise ratio).
- Sample dataset: `examples/mock_rl_dataset.jsonl` (500 rows, pick/clarify/multi_intent).
- Dataset loader/validator: `grpo_intent/dataset.py`, `examples/validate_dataset.py`.

GRPO runner script
- Quick GPU run with DecisionObject rewards: `python scripts/run_decision_grpo.py --model-id gpt2 --dataset examples/mock_rl_dataset.jsonl --steps 50 --batch-size 1 --grad-accum 1 --reward-variant decision_object`
- Bootstrap venv + deps: `bash scripts/bootstrap_trainer_env.sh` (installs transformers/trl/datasets/litellm; install torch separately for your CUDA version).
- Script warns if CUDA isn’t available; add `--no-cuda-check` to skip. Uses TRL (no Unsloth wiring in this script).
