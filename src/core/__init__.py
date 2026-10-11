"""RL 执行模块。Runtime 协调 worker、DataEngine、CheckpointEngine 与 CheckpointStore。
worker 调用现有 mini 引擎；DataEngine 管经验数据，CheckpointEngine 管在线权重，CheckpointStore 管持久化。

RL execution modules. Runtime coordinates workers, DataEngine, CheckpointEngine, and CheckpointStore.
Workers call the existing mini engines; DataEngine handles experience, CheckpointEngine handles online weights, and CheckpointStore handles persistence.
"""
