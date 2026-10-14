"""在线权重快照。保存本轮参数的独立副本与版本。
后续参数更新不会改变已发布的快照。

Online weight snapshot. Keep independent parameter copies and their version.
Later parameter updates do not change the published snapshot.
"""

from ...config import RunConfig
import torch

class Snapshot:
    def __init__(self, config: RunConfig):
        self.config = config

    def snapshot(self, weights: dict[str, torch.Tensor]):
        pass