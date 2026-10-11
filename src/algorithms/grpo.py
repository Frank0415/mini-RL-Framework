import torch

from ..types import TrainBatch


class GRPOAlgorithm:
    def __init__(
        self,
        clip_eps: float = 0.2,
        kl_coef: float = 0.0,
        epsilon: float = 1e-8,
    ):
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.epsilon = epsilon

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
