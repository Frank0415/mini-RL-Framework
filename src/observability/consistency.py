"""检查更新前的训练概率与采样概率是否一致。

ConsistencyChecker 对同一版本、同一词元和同一概率定义计算差异，只统计有效回答位置。
TrainingWorker 在首次参数更新前调用 compare，获得均值、分位数、最大差异和少量样本明细。

Compare training and rollout log probabilities before updates.
ConsistencyChecker compares the same tokens, weight version, and probability definition. It measures only valid response positions.
TrainingWorker calls compare before the first optimizer update. It receives mean, percentile, and maximum differences plus selected sample details.

计划接口 / Planned interfaces:
ConsistencyChecker.compare(batch, old_log_probs)
"""
