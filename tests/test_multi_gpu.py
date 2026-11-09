import copy
import unittest

from mini_megatron import ModelEngine, Trainer
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.algorithms.grpo import GRPOAlgorithm
from src.config import TrainingConfig
from src.core.training_worker.parallel_engine import LogProbModule, ParallelModelEngine
from src.core.training_worker.worker import TrainingWorker

from test_grpo_training import make_batch


def make_model():
    torch.manual_seed(7)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    ))


class LogProbModuleTests(unittest.TestCase):
    def test_forward_preserves_token_probabilities_and_gradients(self):
        model = make_model()
        batch = make_batch()
        actual = LogProbModule(model)(batch.input_ids, batch.attention_mask)
        expected = ModelEngine(model).log_probs(batch.input_ids, batch.attention_mask)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "Two CUDA devices required")
class MultiGPUTrainingTests(unittest.TestCase):
    def test_uneven_minibatches_match_single_gpu_updates(self):
        model = make_model()
        single = ModelEngine(copy.deepcopy(model).to("cuda:1"))
        parallel = ParallelModelEngine(
            model.to("cuda:1"), [torch.device("cuda:1"), torch.device("cuda:0")],
        )
        config = TrainingConfig(learning_rate=0.001, minibatch_size=3, update_epochs=2)
        workers = [
            TrainingWorker(Trainer(engine, learning_rate=config.learning_rate), GRPOAlgorithm(), config)
            for engine in (single, parallel)
        ]
        for worker in workers:
            torch.manual_seed(19)
            metrics = worker.update(make_batch())
            self.assertEqual(worker.actor_version, 4)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual(parallel.device, torch.device("cuda:1"))
        for name, value in parallel.model.state_dict().items():
            torch.testing.assert_close(value, single.model.state_dict()[name], rtol=3e-4, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
