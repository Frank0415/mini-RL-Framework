from collections.abc import Callable
from typing import cast
from uuid import uuid4

from mini_sglang import Engine
from mini_sglang.scheduler.request import RequestStatus

from ...types import Prompt, Sample, SampleGroup


class RolloutWorker:
    def __init__(
        self,
        engine: Engine,
        tokenizer,
        reward_fn: Callable[[Prompt, str], float],
        max_new_tokens: int = 128,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.max_new_tokens = max_new_tokens

    def collect(
        self,
        prompts: list[Prompt],
        samples_per_prompt: int,
    ) -> list[SampleGroup]:
        inputs = [
            prompt.prompt_token_ids
            for prompt in prompts
            for _ in range(samples_per_prompt)
        ]
        results = self.engine.generate_batch(inputs, self.max_new_tokens)
        groups = []
        for index, prompt in enumerate(prompts):
            group_id = uuid4().hex
            responses = results[index * samples_per_prompt:(index + 1) * samples_per_prompt]
            samples = []
            for response in responses:
                if response.status != RequestStatus.FINISHED:
                    raise RuntimeError(f"Rollout failed: {response.finish_reason}")
                text = self.tokenizer.decode(response.output_ids, skip_special_tokens=True)
                samples.append(Sample(
                    sample_id=response.request_id,
                    group_id=group_id,
                    prompt_token_ids=response.input_ids,
                    response_token_ids=response.output_ids,
                    rollout_log_probs=response.output_log_probs,
                    behavior_version=response.weight_version,
                    reward=float(self.reward_fn(prompt, text)),
                    finish_reason=cast(str, response.finish_reason),
                ))
            groups.append(SampleGroup(group_id, prompt.prompt_id, samples))
        return groups
