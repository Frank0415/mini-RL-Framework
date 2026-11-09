"""资源调度。管理 worker 的 CPU/GPU 放置、启动与关闭。

Resource scheduling. Manage worker CPU/GPU placement, startup, and shutdown.
"""

from mini_megatron import ModelEngine
import torch

from ...config import RunConfig
from ..rollout_worker.backends import build_rollout_engine


class ResourceScheduler:
    def __init__(self, config: RunConfig):
        self.config = config
        self.training_devices = [torch.device(device) for device in config.resources.training_devices]
        self.rollout_devices = [torch.device(device) for device in config.resources.rollout_devices]

    def build_training_engine(self) -> ModelEngine:
        return ModelEngine.from_pretrained(
            self.config.model.path,
            device=str(self.training_devices[0]),
            dtype=getattr(torch, self.config.training.dtype),
        )

    def build_rollout_engine(self):
        return build_rollout_engine(
            self.config.model, self.config.rollout, device=str(self.rollout_devices[0]),
        )
