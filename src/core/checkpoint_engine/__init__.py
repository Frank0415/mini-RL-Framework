"""在线权重发布、传输与安装。CheckpointEngine 连接 Snapshot、WeightTransport 和 InstallSession。
Runtime 决定同步时机，并等待推理侧安装与版本提交完成后开始下一轮。

Online weight publication, transfer, and installation. CheckpointEngine connects Snapshot, WeightTransport, and InstallSession.
Runtime schedules synchronization and waits for rollout installation and version commit before starting the next iteration.
"""
