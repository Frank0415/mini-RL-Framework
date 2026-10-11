"""模型评估模块。Evaluator 使用独立评估数据，通过 rollout 组件评估已安装的当前策略。
Runtime 在权重同步后触发评估；评估返回奖励与通过率，保持训练数据位置和随机状态。

Model evaluation. Evaluator uses separate evaluation data and rollout components to assess the installed current policy.
Runtime triggers evaluation after synchronization; evaluation reports rewards and pass rates while preserving training data position and random state.
"""
