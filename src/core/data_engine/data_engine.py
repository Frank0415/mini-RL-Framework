"""把本轮完整回答组整理成训练批次。

DataEngine 接收已评分的完整组，核对采样版本，再补齐序列长度并创建有效词元掩码。
Trainer 放入 RolloutWorker 的结果，再取出 TrainBatch 交给 TrainingWorker。优势值在完整组上计算。

Build a training batch from complete response groups for the current iteration.
DataEngine receives complete scored groups, checks behavior versions, pads sequences, and creates valid-token masks.
Trainer puts rollout results into DataEngine and passes the returned TrainBatch to TrainingWorker. Compute advantages over complete groups.

计划接口 / Planned interfaces:
DataEngine.put(groups), DataEngine.take(n_groups, expected_version)
"""
