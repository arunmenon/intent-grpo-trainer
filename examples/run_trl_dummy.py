"""
Smoke test: run a tiny TRL GRPO training loop on the dummy dataset.

This is for API wiring only; it uses a tiny model (gpt2) and 1 step.
"""

from trl import GRPOTrainer

from examples.dummy_dataset import build_dummy_dataset
from examples.trl_integration import build_trl_grpo_trainer


def main():
    ds = build_dummy_dataset()

    # Use a tiny model for smoke test; swap with your target model.
    trainer: GRPOTrainer = build_trl_grpo_trainer(
        model_name="gpt2",
        train_dataset=ds,
        initial_k=2,
        max_completion_length=32,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        report_to=None,  # disable wandb for smoke test
    )

    trainer.train(max_steps=1)
    print("Completed 1 GRPO step on dummy dataset.")


if __name__ == "__main__":
    main()
