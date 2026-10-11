"""推理权重安装。InstallSession 停止新请求提交，等待已有请求结束，再安装权重、提交版本并恢复提交。
同步首版在 collect 返回后调用 mini-sglang update_weights，由引擎加载权重、清空旧缓存并提交版本。
Runtime 等待安装成功后开始下一轮。

Rollout weight installation. InstallSession stops new submissions, waits for existing requests, installs weights, commits the version, and resumes submissions.
The synchronous version calls mini-sglang update_weights after collect returns; the engine loads weights, clears old caches, and commits the version.
Runtime waits for successful installation before starting the next iteration.
"""
