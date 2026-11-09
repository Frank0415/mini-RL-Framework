"""多设备采样副本。并行分发请求，按输入顺序汇总结果，并向全部副本安装同一版本。

Multi-device rollout replicas. Dispatch requests concurrently, preserve input order, and install the same version on every replica.
"""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from typing import cast

import torch

from ...types import GenerationResult
from .backends import RolloutEngine


class RolloutReplicaGroup:
    def __init__(self, engines: list[RolloutEngine], executor: ThreadPoolExecutor):
        self.engines = engines
        self.executor = executor

    def generate_batch(
        self, input_ids: list[list[int]], max_new_tokens: int,
    ) -> list[GenerationResult]:
        count = len(self.engines)
        pending = [
            (index, self.executor.submit(engine.generate_batch, input_ids[index::count], max_new_tokens))
            for index, engine in enumerate(self.engines)
            if index < len(input_ids)
        ]
        outputs: list[GenerationResult | None] = [None] * len(input_ids)
        for index, future in pending:
            outputs[index::count] = future.result()
        return cast(list[GenerationResult], outputs)

    def update_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], version: int,
    ) -> None:
        weights = tuple(weights)
        pending = [
            self.executor.submit(engine.update_weights, weights, version)
            for engine in self.engines
        ]
        for future in pending:
            future.result()
