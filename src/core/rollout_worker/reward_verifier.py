"""任务评分加载与调用。RewardVerifier 加载实验目录中的 score，供 RolloutWorker 为完成的回答评分。
奖励规则由任务定义；评分结果随 Sample 进入 DataEngine。

Load and call task scoring. RewardVerifier loads the experiment score function for completed rollout responses.
Tasks define reward rules; scores enter DataEngine with Sample records.
"""
