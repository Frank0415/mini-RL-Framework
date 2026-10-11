"""训练触发入口。main 读取 RunConfig，创建 Runtime 并调用 run。
组件初始化、阶段调度和训练循环由 Runtime 与对应 worker 负责。

Training entry point. main reads RunConfig, creates Runtime, and calls run.
Runtime and the corresponding workers own initialization, stage scheduling, and execution.
"""

from config import RunConfig
from core.runtime.runtime import Runtime


def main(config_path: str):
    config = RunConfig.from_toml(config_path)
    runtime = Runtime(config)
    runtime.run()
