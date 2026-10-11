"""张量存放与传递。TransferQueue 管理实际 tensor 的存取和字段读写，供 DataEngine 组装与传递 batch。
选哪些经验用于训练由 Sampler 决定。

Tensor storage and transfer. TransferQueue stores tensors and provides field access for DataEngine batch assembly and delivery.
Sampler decides which experience enters training.
"""
