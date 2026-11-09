import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from src.algorithms.grpo import GRPOAlgorithm
from src.config import ResourceConfig, RunConfig
from src.core.checkpoint_engine.checkpoint_engine import CheckpointEngine
from src.core.data_engine.data_engine import DataEngine
from src.core.rollout_worker.worker import RolloutWorker
from src.core.runtime.resource_scheduler import ResourceScheduler
from src.core.training_worker.evaluation import Evaluator
from src.core.training_worker.worker import TrainingWorker
from src.types import Prompt
from mini_megatron import Trainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT.parent / (
    "mini-sglang/models/smollm2-135m-instruct/"
    "models--HuggingFaceTB--SmolLM2-135M-Instruct/"
    "snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac"
)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "Two CUDA devices required")
class DisaggregatedGPUChecks(unittest.TestCase):
    def setUp(self):
        self.model_dir = Path(os.environ.get("MINI_RL_MODEL_DIR", DEFAULT_MODEL))
        if not (self.model_dir / "config.json").is_file():
            self.skipTest("Set MINI_RL_MODEL_DIR to a local Llama model directory")
        self.config = RunConfig.from_toml(ROOT / "sample/basic_arithmetic")
        self.config.model.path = self.model_dir
        self.config.resources = ResourceConfig(training_devices=["cuda:0"], rollout_devices=["cuda:1"])
        self.config.training.eval_every = 0
        self.config.training.minibatch_size = 4

    def test_separate_training_rollout_sync_and_next_generation(self):
        torch.manual_seed(7)
        with ResourceScheduler(self.config) as scheduler:
            rollout = scheduler.build_rollout_engine()
            actor = scheduler.build_training_engine()
            worker = TrainingWorker(Trainer(actor, learning_rate=0.001), GRPOAlgorithm(), self.config.training)
            rollout_worker = RolloutWorker(
                rollout, SimpleNamespace(decode=lambda ids, **kwargs: "answer"),
                Mock(side_effect=[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]), max_new_tokens=4,
            )
            groups = rollout_worker.collect([Prompt("a", [1, 3, 5], ""), Prompt("b", [1, 4], "")], 2)
            data = DataEngine(self.config)
            before = actor.model.model.norm.weight.detach().clone()
            with patch.object(actor, "export_weights", side_effect=AssertionError("CPU snapshot used")):
                metrics = worker.update(data.to_train_batch(groups))
                CheckpointEngine(self.config).sync(worker, rollout_worker)
                self.assertEqual(rollout.engine.runner.radix_cache.root_node.children, {})
                evaluator = Evaluator(
                    rollout_worker, SimpleNamespace(rows=[1], next_batch=lambda size: [Prompt("eval", [1, 3], "")]),
                )
                self.assertEqual(evaluator.evaluate(worker.export_weights(cpu=False), worker.actor_version)["pass_rate"], 1.0)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
            self.assertFalse(torch.equal(before, actor.model.model.norm.weight))
            self.assertEqual(actor.device, torch.device("cuda:0"))
            self.assertEqual(rollout.device, torch.device("cuda:1"))
            actual = dict(rollout.engine.runner.model.named_parameters(remove_duplicate=False))
            for name, weight in actor.model.state_dict().items():
                torch.testing.assert_close(actual[name], weight.to(device="cuda:1", dtype=actual[name].dtype), rtol=0, atol=0)
            response = rollout_worker.collect([Prompt("next", [1, 3], "")], 1)[0].samples[0]
            self.assertEqual(response.behavior_version, worker.actor_version)
            self.assertEqual(len(response.rollout_log_probs), len(response.response_token_ids))
            print(f"Disaggregated Llama: version={worker.actor_version}, loss={metrics['loss']:.6f}")

    def test_rollout_replicas_install_and_generate_on_both_devices(self):
        self.config.resources.rollout_devices = ["cuda:1", "cuda:0"]
        with ResourceScheduler(self.config) as scheduler:
            rollout = scheduler.build_rollout_engine()
            weights = rollout.engines[0].engine.runner.model.state_dict()
            rollout.update_weights(weights.items(), version=9)
            responses = rollout.generate_batch([[1, 3], [1, 4, 5], [1, 6]], 2)
            self.assertEqual(len(responses), 3)
            self.assertEqual(len({response.request_id for response in responses}), 3)
            self.assertTrue(all(response.weight_version == 9 for response in responses))
            for engine in rollout.engines:
                self.assertEqual(engine.engine.installed_version, 9)
                self.assertEqual(engine.engine.scheduler.requests, {})
                for name, parameter in engine.engine.runner.model.state_dict().items():
                    torch.testing.assert_close(parameter, weights[name].to(engine.device), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
