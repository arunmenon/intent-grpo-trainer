# intent-grpo-trainer

GRPO/TRL scaffold for structured intent disambiguation with JSON schema rewards. Ships with two reward tracks:
- Legacy `final_intents` rewards (R1–R5).
- DecisionObject rewards (JSON decision_type + picked intents + clarifications + reasoning), aligned to PPA-style intent routing.

What’s here
- Reward rubric (R1–R5) for JSON validity/KG paths, decision type, intent match, question coverage, and reasoning keywords. See `grpo_intent/reward_components.py` (legacy) and `grpo_intent/intent_reward.py` (DecisionObject).
- Multi-turn scaffolding for plan-aware rewards: `grpo_intent/plan_schema.py` defines minimal plan/branch/tool types; `grpo_intent/plan_reward.py` implements per-step micro-rewards (slot F1, preconditions/effects, branch/tool validity, latency, policy) and discounted trajectory aggregation.
- Router scaffolding: `grpo_intent/router_schema.py` (RouterInput/Action/Tool types) and `grpo_intent/router_reward.py` (router reward shaping + simulator); TRL wiring in `examples/router_trl_integration.py`.
- TRL integration: `examples/trl_integration.py` (single-turn legacy/DecisionObject) and `examples/plan_trl_integration.py` (plan-aware).
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
3) DecisionObject path (PPA-style, single-turn intent routing):
   - Generate a dataset (template or LLM):  
     ```bash
     # Template-based (fast)
     python3 examples/generate_mock_rl_dataset.py --output examples/mock_rl_dataset.jsonl --count 500 --seed 42 --dedup-prompts --log-every 100
     # LLM-based (set API key first: export LITELLM_API_KEY=...; adjust model/temperature/workers)
     python3 examples/generate_mock_rl_dataset.py --use-llm --llm-model gpt-4o-mini --llm-workers 4 --llm-temperature 0.8 \
       --output examples/mock_rl_dataset.jsonl --count 500 --seed 42 --dedup-prompts --log-every 50
     ```
   - Validate against the KG:  
     ```bash
     python3 examples/validate_dataset.py --data examples/mock_rl_dataset.jsonl --kg examples/synthetic_kg.json
     ```
   - Train with DecisionObject rewards (TRL):  
     ```bash
     # alias with clearer name
     python3 scripts/run_single_turn_grpo.py --model-id gpt2 --dataset examples/mock_rl_dataset.jsonl \
       --steps 50 --batch-size 1 --grad-accum 1 --reward-variant decision_object
     ```
     (Set `CUDA_VISIBLE_DEVICES` as needed; script warns if CUDA is missing. Install torch appropriate to your GPU.)
4) Tune `GRPOConfig` (num_generations, max_completion_length, batch size, bf16/fp16) for your hardware in `examples/trl_integration.py`.

Multi-turn plan rewards (demo)
- The new plan-aware helpers live in `grpo_intent/plan_schema.py` and `grpo_intent/plan_reward.py`. Run a tiny demo:
  ```bash
  python scripts/demo_plan_rewards.py
  ```
- Score trajectories (multi-turn) with plan rewards:
  ```bash
  python scripts/run_plan_grpo.py --demo
  # or supply your own JSONL of plan trajectories
  python scripts/run_plan_grpo.py --dataset /path/to/plan_trajectories.jsonl
  ```
- Plan GRPO training (TRL, scaffold): 
  ```bash
  # Demo dataset; replace with --dataset for your JSONL of gold plan steps
  python scripts/run_plan_trl.py --model-id gpt2 --steps 5 --demo
  ```
- Router GRPO training (TRL, scaffold):
  ```bash
  # Demo dataset; replace with --dataset for your JSONL of router inputs/tool menus
  python scripts/run_router_trl.py --model-id gpt2 --steps 5 --demo
  ```

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
