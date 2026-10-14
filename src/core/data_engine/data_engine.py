"""训练数据采样。选择完整回答组，保留样本的分组关系与采样版本。

Training data sampling. Select complete response groups while preserving sample grouping and behavior versions.
"""
import threading

import torch
from torch.nn.utils.rnn import pad_sequence

from ...config import RunConfig
from ...types import SampleGroup, TrainBatch


class DataEngine:
    def __init__(self, config: RunConfig):
        self.config = config
        self.data_buffer: list[SampleGroup] = []
        self.mutex = threading.Lock()
        self.max_size = 10000000 if not config.data.max_size else config.data.max_size
        self.batch_size = config.training.minibatch_size

    def push(self, sample_groups: list[SampleGroup]):
        with self.mutex:
            if self._len() + len(sample_groups) > self.max_size:
                raise ValueError("Data buffer is full")
            self.data_buffer.extend(sample_groups)

    def _pop(self, n: int) -> list[SampleGroup]:
        ret = self.data_buffer[:n]
        self.data_buffer = self.data_buffer[n:]
        return ret

    def pop(self, n: int) -> list[SampleGroup]:
        with self.mutex:
            return self._pop(n)

    def pop_batch(self) -> list[SampleGroup]:
        with self.mutex:
            return self._pop_batch()

    def to_train_batch(self, sample_groups: list[SampleGroup]) -> TrainBatch:
        samples = [sample for group in sample_groups for sample in group.samples]
        prompt_lengths = torch.tensor([len(sample.prompt_token_ids) for sample in samples])
        response_lengths = torch.tensor([len(sample.response_token_ids) for sample in samples])
        input_ids = pad_sequence([
            torch.tensor(sample.prompt_token_ids + sample.response_token_ids, dtype=torch.long)
            for sample in samples
        ], batch_first=True)
        rollout_log_probs = pad_sequence([
            torch.tensor(sample.rollout_log_probs, dtype=torch.float32)
            for sample in samples
        ], batch_first=True)
        attention_mask = torch.arange(input_ids.shape[1])[None, :] < (
            prompt_lengths + response_lengths
        )[:, None]
        response_mask = torch.arange(rollout_log_probs.shape[1])[None, :] < response_lengths[:, None]
        return TrainBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            rollout_log_probs=rollout_log_probs,
            rewards=torch.tensor([sample.reward for sample in samples], dtype=torch.float32),
            behavior_versions=torch.tensor([sample.behavior_version for sample in samples]),
            sample_ids=[sample.sample_id for sample in samples],
            group_ids=[sample.group_id for sample in samples],
        )

    def _pop_batch(self) -> list[SampleGroup]:
        ret = self.data_buffer
        self.data_buffer = []
        return ret

    def len(self) -> int:
        with self.mutex:
            return self._len()

    def _len(self) -> int:
        return len(self.data_buffer)
