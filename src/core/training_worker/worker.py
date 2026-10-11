"""训练执行入口。TrainingWorker 接收 mini-megatron Trainer 和算法，提供 update、权重导出和状态恢复。
算法安排本批更新；mini-megatron 执行前向、反向传播和优化器步骤，Runtime 协调全局阶段。

Training execution entry point. TrainingWorker receives a mini-megatron Trainer and an algorithm, exposing update, weight export, and recovery.
The algorithm arranges batch updates; mini-megatron executes forward, backward, and optimizer steps; Runtime coordinates overall stages.
"""

from mini_megatron import ModelEngine, Trainer

from ...algorithms.grpo import GRPOAlgorithm
from ...types import TrainBatch


class TrainingWorker:
    def __init__(
        self,
        trainer: Trainer,
        algorithm: GRPOAlgorithm,
        reference: ModelEngine | None = None,
    ):
        self.trainer = trainer
        self.algorithm = algorithm
        self.reference = reference

    @property
    def actor_version(self) -> int:
        return self.trainer.optimizer_step

    def update(self, batch: TrainBatch) -> dict[str, float]:
        return self.algorithm.update(self.trainer, batch, self.reference)

    def export_weights(self):
        return self.trainer.engine.export_weights()

    def state_dict(self) -> dict[str, object]:
        return self.trainer.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.trainer.load_state_dict(state)
