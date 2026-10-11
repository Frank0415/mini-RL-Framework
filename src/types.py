from dataclasses import dataclass, replace

import torch


@dataclass
class Prompt:
    prompt_id: str
    prompt_token_ids: list[int]
    answer: str


@dataclass
class Sample:
    sample_id: str
    group_id: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    rollout_log_probs: list[float]
    behavior_version: int
    reward: float
    finish_reason: str


@dataclass
class SampleGroup:
    group_id: str
    prompt_id: str
    samples: list[Sample]


@dataclass
class TrainBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    response_lengths: torch.Tensor
    rollout_log_probs: torch.Tensor
    rewards: torch.Tensor
    behavior_versions: torch.Tensor
    sample_ids: list[str]
    group_ids: list[str]
    old_log_probs: torch.Tensor | None = None
    advantages: torch.Tensor | None = None

    def to(self, device: torch.device | str):
        return replace(self, **{
            name: value.to(device)
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor)
        })

    def select(self, indices: torch.Tensor):
        values = {}
        selected = indices.tolist()
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                values[name] = value[indices]
            elif isinstance(value, list):
                values[name] = [value[index] for index in selected]
        return replace(self, **values)

    def response_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        mask = self.response_mask.bool()
        positions = self.prompt_lengths[:, None] - 1 + torch.arange(
            mask.shape[1], device=log_probs.device,
        )
        positions = positions.masked_fill(~mask, 0)
        return log_probs.gather(1, positions).masked_fill(~mask, 0)
