"""准备训练数据并更新训练模型。

TrainingWorker 在更新前重算旧策略概率并冻结结果，检查训练与推理差异，再计算优势值。
它持有训练模型、可选参考模型、算法和优化器，按小批次执行前向、反向传播和参数更新。

Prepare training data and update the training model.
TrainingWorker recomputes and freezes old-policy log probabilities before updates, checks rollout consistency, and computes advantages.
It owns the actor, optional reference model, algorithm, and optimizer. It runs forward, backward, and optimizer steps for each minibatch.

计划接口 / Planned interfaces:
TrainingWorker.update(batch), state_dict(), load_state_dict(state)
"""
