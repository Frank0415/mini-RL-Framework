"""训练角色的执行入口。TrainingWorker 初始化并组织训练组件与算法，向 Runtime 提供 update、权重导出和状态恢复。
前向、反向传播与优化器由 mini-megatron 执行；跨 worker 调度由 Runtime 负责。

Execution entry point for training roles. TrainingWorker initializes and connects training components and algorithms for Runtime.
mini-megatron executes forward, backward, and optimizer steps; Runtime coordinates workers and calls update, weight export, and recovery.
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
