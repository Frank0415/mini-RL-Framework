"""根据同题回答的奖励计算 GRPO 训练目标。

GRPOAlgorithm 计算组内优势值和裁剪损失。启用参考模型约束时，再加入 KL 项。
TrainingWorker 传入概率、奖励和掩码，接收损失与指标。模型前向和优化器更新由 TrainingWorker 执行。

Compute the GRPO objective from rewards for responses to the same prompt.
GRPOAlgorithm computes group advantages and the clipped loss. Add a KL term when the reference-model penalty is enabled.
TrainingWorker passes log probabilities, rewards, and masks, then receives the loss and metrics. TrainingWorker runs model forwards and optimizer updates.

计划接口 / Planned interfaces:
GRPOAlgorithm.advantages(groups), GRPOAlgorithm.loss(new_log_probs, batch, ref_log_probs=None)
"""
