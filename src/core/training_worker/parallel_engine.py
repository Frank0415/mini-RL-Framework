"""多卡训练。分发前向计算，在主设备汇总逐 token 概率并归并梯度。

Multi-GPU training. Distribute forwards, gather token log probabilities, and reduce gradients on the primary device.
"""

from mini_megatron import ModelEngine
import torch
from torch import nn


class LogProbModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        return ModelEngine(self.model).log_probs(input_ids, attention_mask)


class ParallelModelEngine(ModelEngine):
    def __init__(self, model: nn.Module, devices: list[torch.device]):
        super().__init__(model)
        self.parallel = nn.DataParallel(
            LogProbModule(model), device_ids=[device.index for device in devices],
        )

    def log_probs(self, input_ids, attention_mask=None):
        return self.parallel(input_ids, attention_mask)
