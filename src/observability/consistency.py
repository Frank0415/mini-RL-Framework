"""训练与推理概率诊断。ConsistencyChecker 对相同 token、版本和概率定义比较两侧 logprob。
TrainingWorker 在参数更新前提供数据，结果交给日志模块；Runtime 关联对应轮次。

Training and rollout probability diagnostics. ConsistencyChecker compares log probabilities for matching tokens, versions, and definitions.
TrainingWorker supplies pre-update data, logging records the results, and Runtime links them to iterations.
"""
