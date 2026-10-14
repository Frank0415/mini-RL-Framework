"""推理后端选择与适配。根据配置创建后端，统一生成结果和权重安装接口。
generate_batch 按输入顺序返回完成结果，失败时报错；update_weights 在安装完成后提交版本。

Rollout backend selection and adaptation. Create a configured backend with common generation results and weight installation interfaces.
generate_batch returns completed results in input order or raises on failure; update_weights commits the version after installation.
"""

from collections.abc import Iterable
from typing import Protocol, cast

import torch

from ...config import ModelConfig, RolloutConfig
from ...types import GenerationResult


class RolloutEngine(Protocol):
    def generate_batch(
        self,
        input_ids: list[list[int]],
        max_new_tokens: int,
    ) -> list[GenerationResult]: ...

    def update_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        version: int,
    ) -> None: ...


class MiniSGLangBackend:
    def __init__(self, model: ModelConfig):
        from mini_sglang import Engine

        self.engine = Engine(model_dir=model.path, do_sample=True)

    def generate_batch(
        self,
        input_ids: list[list[int]],
        max_new_tokens: int,
    ) -> list[GenerationResult]:
        from mini_sglang.scheduler.request import RequestStatus

        results = self.engine.generate_batch(input_ids, max_new_tokens)
        outputs = []
        for response in results:
            if response.status != RequestStatus.FINISHED:
                raise RuntimeError(f"Rollout failed: {response.finish_reason}")
            outputs.append(GenerationResult(
                request_id=response.request_id,
                token_ids=response.output_ids,
                log_probs=response.output_log_probs,
                weight_version=response.weight_version,
                finish_reason=cast(str, response.finish_reason),
            ))
        return outputs

    def update_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        version: int,
    ) -> None:
        self.engine.update_weights(weights, version)


_BACKENDS = {"mini-sglang": MiniSGLangBackend}


def build_rollout_engine(model: ModelConfig, rollout: RolloutConfig) -> RolloutEngine:
    return _BACKENDS[rollout.engine](model)
