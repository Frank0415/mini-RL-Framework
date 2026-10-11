"""模型评估模块。Evaluator 使用独立评估数据和已安装的当前策略，返回奖励与通过率。
Runtime 在权重同步后触发评估；评估保持训练数据位置和随机状态。

Model evaluation. Evaluator uses separate data and the installed current policy to report rewards and pass rates.
Runtime triggers evaluation after synchronization; evaluation preserves training data position and random state.
"""
