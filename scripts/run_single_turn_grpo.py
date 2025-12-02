"""
Alias entrypoint with clearer naming for single-turn intent routing GRPO.

This wraps scripts/run_decision_grpo.py so users can run:
  python scripts/run_single_turn_grpo.py --model-id gpt2 --dataset examples/mock_rl_dataset.jsonl
"""

from run_decision_grpo import main


if __name__ == "__main__":
    main()
