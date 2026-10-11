"""在线权重传输。WeightTransport 将 Snapshot 参数交给推理侧安装过程。
单机可直接传参数；安装、缓存处理和版本提交由 InstallSession 负责。

Online weight transfer. WeightTransport delivers Snapshot parameters to rollout installation.
A local implementation can pass parameters directly; InstallSession owns installation, cache handling, and version commit.
"""
