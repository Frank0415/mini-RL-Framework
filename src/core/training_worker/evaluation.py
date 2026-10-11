"""用独立题目评估当前推理模型。

Evaluator 使用评估数据源和已同步的推理模型，统计奖励和答案通过率。
Trainer 在权重安装成功后按间隔调用 evaluate。评估保持训练数据游标和随机数状态不变。

Evaluate the current rollout model on separate prompts.
Evaluator uses a separate data source and the synchronized rollout model to measure rewards and answer pass rates.
Trainer calls evaluate at configured intervals after weight installation. Evaluation preserves the training data position and random-number states.

计划接口 / Planned interfaces:
Evaluator.evaluate()
"""
