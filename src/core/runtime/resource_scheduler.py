"""资源调度模块。ResourceScheduler 负责 worker 的 CPU/GPU 放置、启动与关闭。
Runtime 决定何时执行各阶段；资源调度负责提供执行条件。

Resource scheduling. ResourceScheduler handles worker CPU/GPU placement, startup, and shutdown.
Runtime decides when each stage runs; resource scheduling supplies its execution resources.
"""
