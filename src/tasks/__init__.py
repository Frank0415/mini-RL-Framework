"""任务与数据源。PromptSource 将实验数据变成 Prompt，供 Runtime 发起采样。
实验目录中的 verifier 定义奖励规则，由 RolloutWorker 使用。

Tasks and data sources. PromptSource converts experiment data into Prompt objects for Runtime to start rollouts.
The experiment verifier defines reward rules used by RolloutWorker.
"""
