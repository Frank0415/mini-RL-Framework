"""批次训练与评估。固定旧概率，计算完整组优势，再按 minibatch 和 epoch 更新参数。
每完成 eval_every 个输入批次评估一次；批次计数随训练状态保存与恢复。

Batch training and evaluation. Freeze old probabilities, compute full-group advantages, and update parameters across minibatches and epochs.
Evaluate after every eval_every input batches; save and restore the batch counter with training state.
"""

from typing import cast

import torch
from mini_megatron import ModelEngine, Trainer

from ...algorithms.grpo import GRPOAlgorithm
from ...types import TrainBatch
from .evaluation import Evaluator
from ...config import TrainingConfig


class TrainingWorker:
    def __init__(
        self,
        trainer: Trainer,
        algorithm: GRPOAlgorithm,
        config: TrainingConfig,
        reference: ModelEngine | None = None,
        evaluator: Evaluator | None = None,
    ):
        if config.eval_every > 0 and evaluator is None:
            raise ValueError("An evaluator is required when eval_every is enabled")
        self.trainer = trainer
        self.algorithm = algorithm
        self.reference = reference
        self.minibatch_size = config.minibatch_size
        self.update_epochs = config.update_epochs
        self.evaluator = evaluator
        self.eval_every = config.eval_every
        self.batches_trained = 0

    @property
    def actor_version(self) -> int:
        return self.trainer.optimizer_step

    def update(self, batch: TrainBatch) -> dict[str, float]:
        if self.algorithm.kl_coef and self.reference is None:
            raise ValueError("A reference model is required for KL loss")
        if self.reference is not None:
            self.reference.model.eval().requires_grad_(False)
        batch = batch.to(self.trainer.engine.device)
        with torch.no_grad():
            batch.old_log_probs = batch.response_log_probs(
                self.trainer.engine.log_probs(batch.input_ids, batch.attention_mask),
            ).detach()
            ref_log_probs = (
                batch.response_log_probs(
                    self.reference.log_probs(batch.input_ids, batch.attention_mask),
                )
                if self.algorithm.kl_coef and self.reference is not None
                else None
            )
        batch.advantages = self.algorithm.advantages(batch.rewards, batch.group_ids)
        totals = {}
        count = 0
        batch_size = batch.input_ids.shape[0]
        for _ in range(self.update_epochs):
            order = torch.randperm(batch_size, device=self.trainer.engine.device)
            for indices in order.split(self.minibatch_size):
                minibatch = batch.select(indices)
                ref_probs = (
                    ref_log_probs[indices] if ref_log_probs is not None else None
                )

                def loss_fn(log_probs, minibatch=minibatch, ref_probs=ref_probs):
                    return self.algorithm.loss(
                        minibatch.response_log_probs(log_probs),
                        minibatch,
                        ref_probs,
                    )

                metrics = self.trainer.train_step(
                    minibatch.input_ids,
                    minibatch.attention_mask,
                    loss_fn,
                )
                for name, value in metrics.items():
                    if name != "optimizer_step":
                        totals[name] = totals.get(name, 0.0) + value * len(indices)
                count += len(indices)

        self.batches_trained += 1
        difference = (batch.old_log_probs - batch.rollout_log_probs).abs()
        metrics = {
            **{name: value / count for name, value in totals.items()},
            "reward_mean": batch.rewards.mean().item(),
            "logprob_abs_diff": difference.masked_select(batch.response_mask.bool())
            .mean()
            .item(),
            "actor_version": self.actor_version,
            "batches_trained": self.batches_trained,
        }
        if (
            self.evaluator is not None
            and self.eval_every > 0
            and self.batches_trained % self.eval_every == 0
        ):
            evaluation = self.evaluator.evaluate(
                self.export_weights(), self.actor_version
            )
            metrics.update(
                {f"eval/{name}": value for name, value in evaluation.items()}
            )
        return metrics

    def export_weights(self):
        return self.trainer.engine.export_weights()

    def state_dict(self) -> dict[str, object]:
        return {**self.trainer.state_dict(), "batches_trained": self.batches_trained}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.trainer.load_state_dict(state)
        self.batches_trained = cast(int, state["batches_trained"])
