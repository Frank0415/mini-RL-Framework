"""推理权重安装过程。InstallSession 组织暂停、安装、提交版本和恢复，确保请求使用完整的一版权重。
mini-sglang 的 update_weights 提供单机加载、旧缓存释放和版本提交；Runtime 等待安装完成。

Rollout weight installation. InstallSession coordinates pause, installation, version commit, and resume so requests use a complete weight version.
mini-sglang update_weights supplies local loading, cache release, and version commit; Runtime waits for completion.
"""
