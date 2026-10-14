"""任务提交调度。管理请求的在途额度，在请求完成后释放额度。

Task submission scheduling. Manage capacity for requests in flight and release it on completion.
"""
from config import RunConfig

class SubmissionScheduler:
    def __init__(self, config: RunConfig):
        self.config = config