import copy
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.algorithms.grpo import GRPOAlgorithm
from mini_megatron import ModelEngine, Trainer
from src.types import TrainBatch


def make_engine():
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    ))
    return ModelEngine(model)


def make_batch():
    ids = torch.tensor([
        [1, 3, 5, 7, 9, 0], [1, 3, 6, 8, 0, 0],
        [1, 4, 7, 10, 12, 14], [1, 4, 7, 11, 13, 0],
    ])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0]])
    return TrainBatch(
        input_ids=ids, attention_mask=ids.ne(0).long(), response_mask=mask,
        prompt_lengths=torch.tensor([2, 2, 3, 3]),
        response_lengths=mask.sum(-1), rollout_log_probs=torch.zeros(4, 3),
        rewards=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        behavior_versions=torch.zeros(4, dtype=torch.long),
        sample_ids=["a0", "a1", "b0", "b1"], group_ids=["a", "a", "b", "b"],
    )


class RecordingGRPO(GRPOAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.old_values = []

    def loss(self, new_log_probs, batch, ref_log_probs=None):
        self.old_values.append({
            sample_id: row.detach().clone()
            for sample_id, row in zip(batch.sample_ids, batch.old_log_probs)
        })
        return super().loss(new_log_probs, batch, ref_log_probs)


class GRPOTrainingTests(unittest.TestCase):
    def test_response_alignment_with_different_prompt_lengths(self):
        batch = make_batch()
        log_probs = torch.arange(20).reshape(4, 5).float()
        actual = batch.response_log_probs(log_probs)
        expected = torch.tensor([[1, 2, 3], [6, 7, 0], [12, 13, 14], [17, 18, 0]]).float()
        torch.testing.assert_close(actual, expected)
        selected = batch.select(torch.tensor([3, 0]))
        self.assertEqual(selected.sample_ids, ["b1", "a0"])
        self.assertEqual(selected.group_ids, ["b", "a"])
        torch.testing.assert_close(selected.input_ids, batch.input_ids[[3, 0]])

    def test_old_probabilities_stay_fixed_across_minibatches_and_epochs(self):
        engine = make_engine()
        batch = make_batch()
        with torch.no_grad():
            old = batch.response_log_probs(engine.log_probs(batch.input_ids, batch.attention_mask))
        expected = dict(zip(batch.sample_ids, old))
        algorithm = RecordingGRPO(minibatch_size=2, update_epochs=2)
        trainer = Trainer(engine, learning_rate=0.01)
        metrics = algorithm.update(trainer, batch)
        self.assertEqual(trainer.optimizer_step, 4)
        self.assertEqual(metrics["actor_version"], 4)
        self.assertEqual(len(algorithm.old_values), 4)
        for minibatch in algorithm.old_values:
            for sample_id, value in minibatch.items():
                torch.testing.assert_close(value, expected[sample_id])
        with torch.no_grad():
            updated = batch.response_log_probs(engine.log_probs(batch.input_ids, batch.attention_mask))
        self.assertFalse(torch.allclose(old, updated))

    def test_reference_model_remains_frozen(self):
        actor, reference = make_engine(), make_engine()
        weights = copy.deepcopy(reference.model.state_dict())
        trainer = Trainer(actor)
        GRPOAlgorithm(kl_coef=0.1).update(trainer, make_batch(), reference=reference)
        for name, parameter in reference.model.named_parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, weights[name])


if __name__ == "__main__":
    unittest.main()
