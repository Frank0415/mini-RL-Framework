"""保存和恢复继续训练所需的状态。

CheckpointStore 保存模型、优化器、训练轮次、权重版本、数据读取位置和随机数状态。
Trainer 在同步轮次结束时保存。恢复后先把训练权重安装到推理模型，再开始下一轮。

Save and restore the state needed to resume training.
CheckpointStore saves the model, optimizer, iteration, weight version, data position, and random-number states.
Trainer saves at synchronous iteration boundaries. After restoring, install actor weights in the rollout model before the next iteration.

计划接口 / Planned interfaces:
CheckpointStore.save(path, state), CheckpointStore.load(path)
"""
