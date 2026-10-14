"""回答采样执行入口。接收推理后端、tokenizer 和评分函数。
collect 生成回答、解码并评分，按题组成 SampleGroup，保留 token、logprob、奖励、版本与结束原因。

Response sampling entry point. Accept a rollout backend, tokenizer, and scoring function.
collect generates, decodes, and scores responses, grouping them into SampleGroup with tokens, log probabilities, rewards, versions, and finish reasons.
"""

from src.types import Prompt


from collections.abc import Callable
from uuid import uuid4

from ...types import Prompt, Sample, SampleGroup
from .backends import RolloutEngine


class RolloutWorker:
    def __init__(
        self,
        engine: RolloutEngine,
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

        for index, prompt in enumerate[Prompt](prompts):
            group_id = uuid4().hex
            responses = results[index * samples_per_prompt:(index + 1) * samples_per_prompt]
            samples = []

            for response in responses:
                text = self.tokenizer.decode(response.token_ids, skip_special_tokens=True)
                samples.append(Sample(
                    sample_id=response.request_id,
                    group_id=group_id,
                    prompt_token_ids=prompt.prompt_token_ids,
                    response_token_ids=response.token_ids,
                    rollout_log_probs=response.log_probs,
                    behavior_version=response.weight_version,
                    reward=float(self.reward_fn(prompt, text)),
                    finish_reason=response.finish_reason,
                ))
            groups.append(SampleGroup(group_id, prompt.prompt_id, samples))

        return groups
