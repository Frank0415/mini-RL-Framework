"""采样提交调度。SubmissionScheduler 管理在途请求额度，并在请求完成后允许继续提交。
由 RolloutWorker 的任务循环使用；完成经验由 DataEngine 管理。

Rollout submission scheduling. SubmissionScheduler limits requests in flight and releases capacity as requests finish.
RolloutWorker uses it within task loops; DataEngine manages completed experience.
"""
