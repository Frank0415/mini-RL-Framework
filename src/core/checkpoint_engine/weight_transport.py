"""在线权重传输。将参数快照与版本传递到推理侧。

Online weight transfer. Deliver parameter snapshots and their version to the rollout side.
"""

from ...config import RunConfig
import torch

class WeightTransport:
    def __init__(self, config: RunConfig):
        self.config = config

    def transport(self, weights: dict[str, torch.Tensor]):
        pass