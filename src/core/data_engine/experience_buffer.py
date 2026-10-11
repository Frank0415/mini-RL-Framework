"""完成经验存储。ExperienceBuffer 保存已经生成并评分的完整回答组，管理容量与版本准入。
DataEngine 写入和取出经验；SubmissionScheduler 管理尚未完成的请求额度。

Completed experience storage. ExperienceBuffer holds complete generated and scored response groups, managing capacity and version admission.
DataEngine inserts and consumes experience; SubmissionScheduler manages capacity for unfinished requests.
"""
