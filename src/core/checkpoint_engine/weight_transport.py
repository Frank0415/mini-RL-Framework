"""在线权重传输。WeightTransport 将 Snapshot 的参数和版本交给推理侧 InstallSession。
单机同步可直接传参数；InstallSession 负责安装，mini-sglang 负责加载与缓存处理。

Online weight transfer. WeightTransport delivers Snapshot parameters and version to the rollout InstallSession.
Local synchronization can pass parameters directly; InstallSession coordinates installation, and mini-sglang handles loading and caches.
"""
