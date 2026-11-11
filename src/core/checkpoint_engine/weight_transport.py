"""在线权重传输。将参数快照与版本传递到推理侧。

Online weight transfer. Deliver parameter snapshots and their version to the rollout side.
"""

from collections.abc import Iterable, Iterator

import torch


class WeightTransport:
    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)

    def transport(
        self, weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            weight = weight.detach()
            if self.device.type == "cuda" and weight.device.type == "cpu":
                weight = weight.pin_memory()
            yield name, weight.to(self.device, non_blocking=self.device.type == "cuda")
