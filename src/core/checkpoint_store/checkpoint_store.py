"""训练状态持久化入口。CheckpointStore 组织 store_checkpoint 与 restore_checkpoint 的磁盘读写。
Runtime 在轮次边界保存和恢复状态，恢复后通过 CheckpointEngine 同步推理模型。

Training-state persistence entry point. CheckpointStore coordinates disk writes and reads through store_checkpoint and restore_checkpoint.
Runtime saves and restores at iteration boundaries, then synchronizes rollout through CheckpointEngine.
"""
