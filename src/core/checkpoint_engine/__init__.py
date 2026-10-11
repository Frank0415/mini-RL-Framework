"""在线权重发布、传输与安装。CheckpointEngine 通过 WeightSync 入口组织快照、参数传输和推理侧安装。
Runtime 协调同步时机，CheckpointStore 负责持久化与恢复。

Online weight publication, transfer, and installation. CheckpointEngine coordinates these through the WeightSync entry point.
Runtime schedules synchronization; CheckpointStore handles persistence and recovery.
"""
