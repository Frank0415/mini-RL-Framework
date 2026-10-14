"""推理权重安装。停止新请求提交，等待已有请求结束，安装权重并清空旧缓存。
安装完成后提交新版本，再恢复请求提交。

Rollout weight installation. Stop new submissions, wait for existing requests, install weights, and clear old caches.
Commit the new version after installation, then resume submissions.
"""

from ...config import RunConfig
import torch

class InstallSession:
    def __init__(self, config: RunConfig):
        self.config = config

    def install(self, weights: dict[str, torch.Tensor]):
        pass