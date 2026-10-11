"""采样提交调度。SubmissionScheduler 管理任务请求的在途额度，完成后释放额度。
RolloutWorker 的任务循环使用它；mini-sglang 负责模型批次调度，DataEngine 负责完成经验。

Rollout submission scheduling. SubmissionScheduler manages capacity for task requests in flight and releases it on completion.
RolloutWorker uses it within task loops; mini-sglang schedules model batches, and DataEngine handles completed experience.
"""
