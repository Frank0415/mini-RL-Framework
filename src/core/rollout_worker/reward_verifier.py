"""任务评分加载与调用。RewardVerifier 从配置路径加载实验 verifier 的 score 函数。
RolloutWorker 用题目标签和回答文本评分；奖励随 Sample 进入 DataEngine。

Task scoring loader. RewardVerifier loads the experiment verifier's score function from the configured path.
RolloutWorker scores task labels and response text; rewards enter DataEngine with Sample records.
"""
