# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `grpo_intent/`: reward components (`reward_components.py`), parsing/schema helpers (`schema.py`), rollout scheduling (`schedule.py`), replay cache (`replay_cache.py`), and a scaffolded trainer (`trainer_scaffold.py`).
- Examples sit in `examples/`: dummy dataset builder (`dummy_dataset.py`), TRL integration (`trl_integration.py`), smoke-test entrypoint (`run_trl_dummy.py`), and a hook template (`hook_template.py`) if you plug GRPO into a custom loop.
- Keep new modules small and focused; prefer adding helpers under `grpo_intent/` rather than expanding example scripts.

## Build, Test, and Development Commands
- Install minimal deps (local, no GPU assumptions): `pip install -U transformers trl datasets`.
- Smoke-test the pipeline with a tiny model/dataset: `python examples/run_trl_dummy.py` (runs 1 GRPO step on the dummy data).
- For interactive tweaks, import `build_trl_grpo_trainer` from `examples/trl_integration.py` and pass your dataset with the expected columns (`gold_intents`, `gold_decision`, `ambiguous_slots`, `reasoning_terms`, optional `known_paths`).

## Coding Style & Naming Conventions
- Python-first, PEP8 spacing (4 spaces) with type hints and dataclasses where appropriate.
- Use snake_case for functions/variables, PascalCase for classes, and keep function arguments explicit rather than relying on globals.
- Prefer small, pure functions; add short doctrings when behavior is non-obvious (reward math, parsing rules).
- Follow existing patterns for wrappers that pull labels from `**kwargs` so TRL/Unsloth trainers can route batch columns correctly.

## Testing Guidelines
- Primary check is the smoke test: run `python examples/run_trl_dummy.py` after changes to rewards, parsing, or scheduling.
- When adding reward logic, craft a tiny in-memory dataset mirroring `examples/dummy_dataset.py` and assert expected scores in an ad-hoc script before wiring to trainers.
- If you introduce optional dependencies (e.g., new model backends), guard imports and provide a fallback or clear error message.

## Commit & Pull Request Guidelines
- Use clear, imperative commit messages describing scope (e.g., `Add coverage penalty to R4`).
- In PRs, include: summary of behavior changes, commands run (e.g., smoke test), and any dataset/model assumptions.
- Add short inline comments only when logic is subtle (numerical stability, clipping, cache mixing).
- Avoid committing large artifacts (checkpoints, datasets); reference how to regenerate instead.
