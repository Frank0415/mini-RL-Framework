"""训练状态持久化。CheckpointStore 在一个类中提供 save 和 load，读写模型、优化器、随机状态、数据位置与运行进度。
Runtime 决定保存和恢复时机，恢复后通过 CheckpointEngine 安装推理权重。

Training-state persistence. CheckpointStore provides save and load in one class for models, optimizers, random state, data position, and run progress.
Runtime schedules saving and recovery, then installs rollout weights through CheckpointEngine after restoration.
"""
