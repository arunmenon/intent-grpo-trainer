"""
Hook template to connect IntentGRPOTrainer to a real model/trainer.

This avoids guessing Unsloth/TRL internals. Replace the TODOs with the exact
calls from your version of Unsloth/TRL. The generate hook uses standard
Transformers .generate so it will run anywhere (B200, H100, single GPU/CPU),
but you can swap in Unsloth's fast generate if available.
"""

from typing import Callable, List, Sequence, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_intent.trainer_scaffold import IntentGRPOTrainer, IntentTrainerConfig, TrainerHooks


def make_generate_fn(
    model,
    tokenizer,
    *,
    use_unsloth_fast: bool = False,
    sampling_params_factory: Optional[Callable[[int, float], dict]] = None,
) -> Callable[[str, int, int, float], List[str]]:
    """
    Returns a callable that samples `num` completions for one prompt.

    - Default: vanilla HF generate (works everywhere).
    - Fast path: if use_unsloth_fast=True and your model exposes a vLLM/Unsloth
      generation hook, you can pass a sampling_params_factory returning kwargs.

    Note: To avoid guessing APIs, we keep the unsloth fast path as an opt-in; wire
    it to your version of FastLanguageModel/vLLM (e.g., using SamplingParams).
    """

    if use_unsloth_fast and sampling_params_factory is not None and hasattr(model, "generate"):
        # User-provided fast params factory; assumes model.generate can accept them.
        def _fast_generate(prompt: str, num: int, max_len: int, temperature: float) -> List[str]:
            if num <= 0:
                return []
            params = sampling_params_factory(max_len, temperature)  # e.g., {"max_tokens": max_len, "temperature": temperature}
            # Adjust num_return_sequences if your backend requires it.
            params.setdefault("num_return_sequences", num)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **params)
            completion_tokens = outputs[:, inputs["input_ids"].shape[1] :]
            return tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        return _fast_generate

    def _generate(prompt: str, num: int, max_len: int, temperature: float) -> List[str]:
        if num <= 0:
            return []
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_len,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                num_return_sequences=num,
            )
        # Slice off the prompt tokens and decode completions.
        completion_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        return tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

    return _generate


def make_backward_update_fn(grpo_step_fn: Callable[[Sequence[str], List[List[str]], List[float]], None]):
    """
    Adapts your GRPO trainer's update step to the scaffold.

    - grpo_step_fn should accept (prompts, completion_groups, advantages).
      Connect this to the official Unsloth/TRL GRPO trainer update method you use.
    """

    def _backward_update(prompts, completion_groups, advantages):
        return grpo_step_fn(prompts, completion_groups, advantages)

    return _backward_update


def build_trainer(model_name: str, config: IntentTrainerConfig, grpo_step_fn) -> IntentGRPOTrainer:
    """
    Convenience builder: loads a HF model/tokenizer, wraps generate/backward hooks,
    and returns an IntentGRPOTrainer. Swap model loading for Unsloth if desired.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    hooks = TrainerHooks(
        generate_fn=make_generate_fn(model, tokenizer),
        backward_update_fn=make_backward_update_fn(grpo_step_fn),
    )
    return IntentGRPOTrainer(config=config, hooks=hooks)


# Example usage (pseudo-code; wire your own grpo_step_fn and data):
# config = IntentTrainerConfig()
# trainer = build_trainer("gpt2", config, grpo_step_fn=my_grpo_update)
# batch_prompts = [...]
# prompt_texts = [...]
# gold_intents = [...]
# gold_decisions = [...]
# ambiguous_slots = [...]
# reasoning_terms = [...]
# info = trainer.train_step(
#     step=0,
#     prompts=batch_prompts,
#     prompt_texts=prompt_texts,
#     gold_intents=gold_intents,
#     gold_decisions=gold_decisions,
#     ambiguous_slots=ambiguous_slots,
#     reasoning_terms=reasoning_terms,
# )
# print(info)
