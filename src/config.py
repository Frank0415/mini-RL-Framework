"""读取同步 GRPO 的运行配置。

RunConfig 保存模型路径、数据路径、设备、训练参数和日志目录。
main 从 TOML 文件读取配置，再把所需参数传给各模块。

Read the configuration for synchronous GRPO.
RunConfig holds model paths, data paths, devices, training settings, and the log directory.
main reads the TOML file and passes the required settings to each module.

计划接口 / Planned interfaces:
RunConfig.from_toml(path)
"""
