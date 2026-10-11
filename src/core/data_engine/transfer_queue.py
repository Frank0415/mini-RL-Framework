"""后续跨进程传递训练张量。TransferQueue 存放张量并支持按字段读取。
同步版本直接传递 TrainBatch 对象。

Transfer training tensors between processes in a future version. TransferQueue stores tensors and supports field reads.
The synchronous version passes TrainBatch objects directly.
"""
