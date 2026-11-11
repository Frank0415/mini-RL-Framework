"""资源调度。管理 worker 的 CPU/GPU 放置、启动与关闭。

Resource scheduling. Manage worker CPU/GPU placement, startup, and shutdown.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

from mini_megatron import ModelEngine
import torch

from ...config import RunConfig
from ..rollout_worker.backends import build_rollout_engine
from ..rollout_worker.replica_group import RolloutReplicaGroup
from ..training_worker.parallel_engine import ParallelModelEngine


class ResourceScheduler:
    def __init__(self, config: RunConfig):
        self.config = config
        self.training_devices = [torch.device(device) for device in config.resources.training_devices]
        self.rollout_devices = [torch.device(device) for device in config.resources.rollout_devices]
        self.workers = ExitStack()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.workers.__exit__(*args)

    def build_training_engine(self) -> ModelEngine:
        engine = ModelEngine.from_pretrained(
            self.config.model.path,
            device=str(self.training_devices[0]),
            dtype=getattr(torch, self.config.training.dtype),
        )
        if len(self.training_devices) > 1:
            engine = ParallelModelEngine(engine.model, self.training_devices)
        return engine

    def build_rollout_engine(self):
        engines = [
            build_rollout_engine(self.config.model, self.config.rollout, device=str(device))
            for device in self.rollout_devices
        ]
        if len(engines) == 1:
            return engines[0]
        executor = self.workers.enter_context(ThreadPoolExecutor(max_workers=len(engines)))
        return RolloutReplicaGroup(engines, executor)
