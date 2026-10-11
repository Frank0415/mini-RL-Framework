"""经验张量存放与传递。TransferQueue 提供 tensor 存取与字段读写，供 DataEngine 组装和传递 batch。
Sampler 决定训练使用哪些经验；模型参数传输属于 CheckpointEngine。

Experience tensor storage and transfer. TransferQueue provides tensor storage and field access for DataEngine batch assembly and delivery.
Sampler selects training experience; CheckpointEngine handles model parameter transfer.
"""
