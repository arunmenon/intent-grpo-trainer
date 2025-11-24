# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `grpo_intent/`: reward components (`reward_components.py`), parsing/schema helpers (`schema.py`), rollout scheduling (`schedule.py`), replay cache (`replay_cache.py`), and a scaffolded trainer (`trainer_scaffold.py`).
- Examples sit in `examples/`: dummy dataset builder (`dummy_dataset.py`), TRL integration (`trl_integration.py`), smoke-test entrypoint (`run_trl_dummy.py`), and a hook template (`hook_template.py`) if you plug GRPO into a custom loop.
- Keep new modules small and focused; add helpers under `grpo_intent/` and keep examples minimal.

## Architecture & Key Classes
- `IntentGRPOTrainer` (`trainer_scaffold.py`) mixes cache + fresh generations, applies R1–R5 rewards, normalizes advantages, and calls your hooks.
- `TrainerHooks` wraps `generate_fn` and `backward_update_fn`; treat prompts/kwargs as opaque user data from your loader.
- `IntentTrainerConfig` holds reward weights, cache mix ratio, and rollout `schedule`; `RolloutSchedule.at(step)` returns `(k, max_len, temp)`.
- `ExperienceCache` keeps per-prompt top-K completions; `sample_mix` returns cached strings plus how many fresh to generate.
- `RewardSettings` + R1–R5 in `reward_components.py`; JSON parsing and intent validation live in `ParsedOutput`/`try_parse_completion`/`validate_intents_exist` (`schema.py`).

## Build, Test, and Development Commands
- Install deps: `pip install -U transformers trl datasets`.
- Smoke test: `python examples/run_trl_dummy.py` (1 GRPO step on dummy data).
- For interactive tweaks, import `build_trl_grpo_trainer` from `examples/trl_integration.py` and pass a dataset with `gold_intents`, `gold_decision`, `ambiguous_slots`, `reasoning_terms`, optional `known_paths`.

## Coding Style & Naming Conventions
- Python-first, PEP8 spacing (4 spaces) with type hints and dataclasses when helpful.
- Use snake_case for functions/variables, PascalCase for classes; keep function args explicit rather than relying on globals.
- Prefer small, pure functions; add short docstrings when behavior is non-obvious (reward math, parsing rules).
- Follow existing patterns for wrappers that pull labels from `**kwargs` so TRL/Unsloth trainers can route batch columns correctly.

## Testing Guidelines
- Primary check: run `python examples/run_trl_dummy.py` after changes to rewards, parsing, or scheduling.
- When adding reward logic, craft a tiny in-memory dataset mirroring `examples/dummy_dataset.py` and assert expected scores in a quick script before wiring to trainers.
- If you introduce optional dependencies (e.g., new model backends), guard imports and provide a fallback or clear error message.

## Commit & Pull Request Guidelines
- Use clear, imperative commit messages describing scope (e.g., `Add coverage penalty to R4`).
- In PRs, include: summary of behavior changes, commands run (smoke test), and dataset/model assumptions.
- Add short inline comments only when logic is subtle (numerical stability, clipping, cache mixing).
- Avoid committing large artifacts (checkpoints, datasets); reference how to regenerate instead.
