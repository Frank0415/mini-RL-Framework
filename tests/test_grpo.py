import math
import unittest

import torch

from src.algorithms.grpo import GRPOAlgorithm
from src.types import TrainBatch


def make_batch(mask, advantages):
    mask = torch.tensor(mask, dtype=torch.bool)
    size, length = mask.shape
    return TrainBatch(
        input_ids=torch.zeros(size, length + 1, dtype=torch.long),
        attention_mask=torch.ones(size, length + 1, dtype=torch.long),
        response_mask=mask,
        prompt_lengths=torch.ones(size, dtype=torch.long),
        response_lengths=mask.sum(-1),
        rollout_log_probs=torch.zeros(size, length),
        rewards=torch.zeros(size),
        behavior_versions=torch.zeros(size, dtype=torch.long),
        sample_ids=[str(i) for i in range(size)],
        group_ids=["group"] * size,
        old_log_probs=torch.zeros(size, length, requires_grad=True),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )


class GRPOTests(unittest.TestCase):
    def test_advantages_use_separate_groups_and_handle_equal_rewards(self):
        rewards = torch.tensor([1.0, 4.0, 0.0, 4.0], requires_grad=True)
        actual = GRPOAlgorithm().advantages(rewards, ["a", "b", "a", "b"])
        torch.testing.assert_close(actual, torch.tensor([1.0, 0.0, -1.0, 0.0]))
        self.assertFalse(actual.requires_grad)

    def test_first_update_has_gradient_at_ratio_one(self):
        batch = make_batch([[1], [1]], [1, -1])
        new = torch.zeros(2, 1, requires_grad=True)
        loss, _ = GRPOAlgorithm().loss(new, batch)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(new.grad, torch.tensor([[-0.5], [0.5]]))
        self.assertIsNone(batch.old_log_probs.grad)

    def test_clipping_stops_excessive_updates_in_both_directions(self):
        batch = make_batch([[1], [1]], [1, -1])
        new = torch.tensor([[math.log(1.5)], [math.log(0.5)]], requires_grad=True)
        loss, metrics = GRPOAlgorithm().loss(new, batch)
        loss.backward()
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        torch.testing.assert_close(new.grad, torch.zeros_like(new))
        self.assertEqual(metrics["clip_fraction"], 1)

    def test_response_average_preserves_equal_answer_weight(self):
        batch = make_batch([[1, 0], [1, 1]], [1, -1])
        new = torch.tensor([[0.0, float("inf")], [0.0, 0.0]], requires_grad=True)
        loss, _ = GRPOAlgorithm().loss(new, batch)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(new.grad, torch.tensor([[-0.5, 0.0], [0.25, 0.25]]))

    def test_reference_kl_is_differentiable_only_through_current_policy(self):
        batch = make_batch([[1]], [0])
        new = torch.tensor([[math.log(2)]], requires_grad=True)
        reference = torch.zeros(1, 1, requires_grad=True)
        loss, _ = GRPOAlgorithm(kl_coef=0.2).loss(new, batch, reference)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.2 * (math.log(2) - 0.5), places=6)
        torch.testing.assert_close(new.grad, torch.tensor([[0.1]]))
        self.assertIsNone(reference.grad)


if __name__ == "__main__":
    unittest.main()
