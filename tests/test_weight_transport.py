from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from mini_megatron import ModelEngine, Trainer
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.algorithms.grpo import GRPOAlgorithm
from src.config import RunConfig, TrainingConfig
from src.core.checkpoint_engine.checkpoint_engine import CheckpointEngine
from src.core.checkpoint_engine.weight_transport import WeightTransport
from src.core.training_worker.worker import TrainingWorker


ROOT = Path(__file__).resolve().parents[1]


class WeightTransportTests(unittest.TestCase):
    def test_same_device_transfer_is_lazy_and_shares_storage(self):
        consumed = []
        weights = [torch.ones(2, requires_grad=True), torch.zeros(3)]

        def source():
            for index, weight in enumerate(weights):
                consumed.append(index)
                yield str(index), weight

        stream = WeightTransport("cpu").transport(source())
        self.assertEqual(consumed, [])
        name, value = next(stream)
        self.assertEqual(name, "0")
        self.assertEqual(consumed, [0])
        self.assertEqual(value.data_ptr(), weights[0].data_ptr())
        self.assertFalse(value.requires_grad)
        self.assertEqual(next(stream)[1].data_ptr(), weights[1].data_ptr())

    def test_online_sync_uses_live_weights_and_snapshot_export_stays_independent(self):
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        ))
        worker = TrainingWorker(Trainer(ModelEngine(model)), GRPOAlgorithm(), TrainingConfig())
        snapshot = worker.export_weights()
        installed = {}
        backend = SimpleNamespace(update_weights=lambda weights, version: installed.update(dict(weights)))
        config = RunConfig.from_toml(ROOT / "sample/basic_arithmetic")
        worker.trainer.engine.export_weights = Mock(side_effect=AssertionError("CPU snapshot used"))
        CheckpointEngine(config).sync(worker, SimpleNamespace(engine=backend))
        for name, parameter in model.state_dict().items():
            self.assertEqual(installed[name].data_ptr(), parameter.data_ptr())
            self.assertNotEqual(snapshot[name].data_ptr(), parameter.data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cpu_to_cuda_and_cuda_to_cpu_preserve_values(self):
        weight = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        gpu = dict(WeightTransport("cuda:0").transport([("weight", weight)]))["weight"]
        torch.testing.assert_close(gpu.cpu(), weight)
        host = dict(WeightTransport("cpu").transport([("weight", gpu)]))["weight"]
        torch.testing.assert_close(host, weight)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two CUDA devices required")
    def test_cuda_to_cuda_preserves_values_on_destination(self):
        weight = torch.arange(21, dtype=torch.float32, device="cuda:1").reshape(3, 7)
        transferred = dict(WeightTransport("cuda:0").transport([("weight", weight)]))["weight"]
        self.assertEqual(transferred.device, torch.device("cuda:0"))
        torch.testing.assert_close(transferred.cpu(), weight.cpu())


if __name__ == "__main__":
    unittest.main()
