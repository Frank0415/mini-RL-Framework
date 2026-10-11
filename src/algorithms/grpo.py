"""GRPO 算法。计算组内优势值、概率比例、裁剪损失和可选 reference KL。
update 固定本批旧概率并安排 minibatch 更新，通过 mini-megatron 执行训练；Runtime 协调采样与同步。

GRPO algorithm. Compute group advantages, probability ratios, clipped loss, and optional reference KL.
update freezes old probabilities and schedules minibatches through mini-megatron; Runtime coordinates rollout and synchronization.
"""

import torch
from mini_megatron import ModelEngine, Trainer

from ..types import TrainBatch


class GRPOAlgorithm:
    def __init__(
        self,
        clip_eps: float = 0.2,
        kl_coef: float = 0.0,
        epsilon: float = 1e-8,
        minibatch_size: int = 4,
        update_epochs: int = 1,
    ):
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.epsilon = epsilon
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs

    def advantages(
        self,
        rewards: torch.Tensor,
        group_ids: list[str],
    ) -> torch.Tensor:
        rewards = rewards.detach().float()
        advantages = torch.zeros_like(rewards)
        groups = {}
        for index, group_id in enumerate(group_ids):
            groups.setdefault(group_id, []).append(index)
        for indices in groups.values():
            values = rewards[indices]
            advantages[indices] = (values - values.mean()) / (
                values.std(correction=0) + self.epsilon
            )
        return advantages

    def loss(
        self,
        new_log_probs: torch.Tensor,
        batch: TrainBatch,
        ref_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mask = batch.response_mask.bool()
        lengths = mask.sum(dim=-1).clamp_min(1)
        log_ratio = (new_log_probs - batch.old_log_probs.detach()).masked_fill(~mask, 0)
        ratio = log_ratio.exp()
        advantage = batch.advantages.detach()[:, None]
        objective = torch.minimum(
            ratio * advantage,
            ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantage,
        )
        policy_loss = -objective.masked_fill(~mask, 0).sum(dim=-1) / lengths
        kl = torch.zeros_like(new_log_probs)
        if self.kl_coef:
            if ref_log_probs is None:
                raise ValueError("Reference log probabilities are required for KL loss")
            ref_ratio = (ref_log_probs.detach() - new_log_probs).masked_fill(~mask, 0)
            kl = ref_ratio.exp() - ref_ratio - 1
        kl_mean = kl.masked_fill(~mask, 0).sum(dim=-1) / lengths
        loss = (policy_loss + self.kl_coef * kl_mean).mean()
        clipped = (ratio - 1).abs() > self.clip_eps
        return loss, {
            "policy_loss": policy_loss.mean().detach().item(),
            "kl": kl_mean.mean().detach().item(),
            "clip_fraction": clipped.masked_select(mask).float().mean().item(),
        }

    def update(
        self,
        trainer: Trainer,
        batch: TrainBatch,
        reference: ModelEngine | None = None,
    ) -> dict[str, float]:
        if self.kl_coef and reference is None:
            raise ValueError("A reference model is required for KL loss")
        if reference is not None:
            reference.model.eval().requires_grad_(False)
        batch = batch.to(trainer.engine.device)
        with torch.no_grad():
            batch.old_log_probs = batch.response_log_probs(
                trainer.engine.log_probs(batch.input_ids, batch.attention_mask),
            ).detach()
            ref_log_probs = (
                batch.response_log_probs(reference.log_probs(batch.input_ids, batch.attention_mask))
                if self.kl_coef else None
            )
        batch.advantages = self.advantages(batch.rewards, batch.group_ids)
        totals = {}
        count = 0
        batch_size = batch.input_ids.shape[0]
        for _ in range(self.update_epochs):
            order = torch.randperm(batch_size, device=trainer.engine.device)
            for indices in order.split(self.minibatch_size):
                minibatch = batch.select(indices)
                ref_probs = ref_log_probs[indices] if ref_log_probs is not None else None

                def loss_fn(log_probs, minibatch=minibatch, ref_probs=ref_probs):
                    return self.loss(
                        minibatch.response_log_probs(log_probs), minibatch, ref_probs,
                    )

                metrics = trainer.train_step(
                    minibatch.input_ids, minibatch.attention_mask, loss_fn,
                )
                for name, value in metrics.items():
                    if name != "optimizer_step":
                        totals[name] = totals.get(name, 0.0) + value * len(indices)
                count += len(indices)
        difference = (batch.old_log_probs - batch.rollout_log_probs).abs()
        return {
            **{name: value / count for name, value in totals.items()},
            "reward_mean": batch.rewards.mean().item(),
            "logprob_abs_diff": difference.masked_select(batch.response_mask.bool()).mean().item(),
            "actor_version": trainer.optimizer_step,
        }
