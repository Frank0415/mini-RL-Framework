from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from src.config import ResourceConfig, RunConfig
from src.core.runtime.resource_scheduler import ResourceScheduler


ROOT = Path(__file__).resolve().parents[1]


class ResourceSchedulerTests(unittest.TestCase):
    def test_existing_config_keeps_single_gpu_placement(self):
        content = (ROOT / "sample/basic_arithmetic/config.toml").read_text()
        content = content.replace('\n[resources]\ntraining_devices = ["cuda:0"]\nrollout_devices = ["cuda:0"]\n', '')
        with TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.toml"
            config_file.write_text(content)
            config = RunConfig.from_toml(config_file)
        self.assertEqual(config.resources.training_devices, ["cuda:0"])
        self.assertEqual(config.resources.rollout_devices, ["cuda:0"])

    def test_toml_separates_training_and_rollout_devices(self):
        content = (ROOT / "sample/basic_arithmetic/config.toml").read_text()
        with TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.toml"
            config_file.write_text(content.replace(
                'training_devices = ["cuda:0"]\nrollout_devices = ["cuda:0"]',
                'training_devices = ["cuda:2", "cuda:0"]\nrollout_devices = ["cuda:1"]',
            ))
            config = RunConfig.from_toml(config_file)
        self.assertEqual(config.resources.training_devices, ["cuda:2", "cuda:0"])
        self.assertEqual(config.resources.rollout_devices, ["cuda:1"])

    def test_builders_pass_explicit_device_and_training_precision(self):
        config = RunConfig.from_toml(ROOT / "sample/basic_arithmetic")
        config.resources = ResourceConfig(training_devices=["cpu"], rollout_devices=["cuda:3"])
        config.training.dtype = "bfloat16"
        scheduler = ResourceScheduler(config)
        with (
            patch("src.core.runtime.resource_scheduler.ModelEngine.from_pretrained") as training,
            patch("src.core.runtime.resource_scheduler.build_rollout_engine") as rollout,
        ):
            self.assertIs(scheduler.build_training_engine(), training.return_value)
            self.assertIs(scheduler.build_rollout_engine(), rollout.return_value)
        training.assert_called_once_with(config.model.path, device="cpu", dtype=torch.bfloat16)
        rollout.assert_called_once_with(config.model, config.rollout, device="cuda:3")

    def test_multiple_training_devices_wrap_the_primary_model(self):
        config = RunConfig.from_toml(ROOT / "sample/basic_arithmetic")
        config.resources.training_devices = ["cuda:2", "cuda:0"]
        with (
            patch("src.core.runtime.resource_scheduler.ModelEngine.from_pretrained") as training,
            patch("src.core.runtime.resource_scheduler.ParallelModelEngine") as parallel,
            ResourceScheduler(config) as scheduler,
        ):
            self.assertIs(scheduler.build_training_engine(), parallel.return_value)
        parallel.assert_called_once_with(training.return_value.model, [torch.device("cuda:2"), torch.device("cuda:0")])


if __name__ == "__main__":
    unittest.main()
