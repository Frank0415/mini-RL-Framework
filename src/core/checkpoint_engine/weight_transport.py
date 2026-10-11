"""后续远程权重同步中传输参数。WeightTransport 负责把快照送到推理进程。
同步版本由 WeightSync 直接把快照交给推理引擎。

Transfer parameters during future remote weight synchronization. WeightTransport sends snapshots to the rollout process.
The synchronous WeightSync passes snapshots directly to the rollout engine.
"""
