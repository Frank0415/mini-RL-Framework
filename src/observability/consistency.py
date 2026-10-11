"""训练与推理概率诊断。ConsistencyChecker 比较相同 token、权重版本和概率定义下的两侧 logprob。
训练侧概率在参数更新前重算，与 Sample 保留的 rollout_log_probs 对齐；结果交给 RunLogger。

Training and rollout probability diagnostics. ConsistencyChecker compares log probabilities for matching tokens, weight versions, and probability definitions.
Training probabilities are recomputed before updates and aligned with Sample rollout_log_probs; RunLogger records the results.
"""
