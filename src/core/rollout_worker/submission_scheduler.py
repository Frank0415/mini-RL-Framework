"""后续并发采样中控制在途请求数量。SubmissionScheduler 在请求完成后释放提交额度。
同步版本由 Trainer 等待本轮采样完成后再训练。

Limit requests in progress during future concurrent rollout. SubmissionScheduler releases submission slots when requests finish.
The synchronous Trainer waits for the current rollout to finish before training.
"""
