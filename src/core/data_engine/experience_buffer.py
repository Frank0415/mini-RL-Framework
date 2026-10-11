"""完成经验存储。ExperienceBuffer 保存已经生成并评分的回答组，管理容量和版本准入。
DataEngine 写入与取出经验；在途请求额度由 SubmissionScheduler 管理。

Completed experience storage. ExperienceBuffer holds generated and scored response groups and manages capacity and version admission.
DataEngine inserts and consumes experience; SubmissionScheduler manages requests in flight.
"""
