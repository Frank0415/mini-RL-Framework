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
