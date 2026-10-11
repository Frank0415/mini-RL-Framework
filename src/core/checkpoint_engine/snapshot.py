"""后续权重同步拆分时保存权重快照的数据结构。
当前 WeightSnapshot 定义在 src.core.weight_sync，保存独立复制的权重和版本。

A future location for weight snapshot data when synchronization is split into separate modules.
WeightSnapshot currently belongs in src.core.weight_sync and holds copied weights and their version.
"""
