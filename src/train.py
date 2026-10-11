"""训练触发入口。main 读取 RunConfig，创建 Runtime 并调用 run。
Runtime 负责组件连接和同步训练循环，worker 执行各自的阶段。

Training entry point. main reads RunConfig, creates Runtime, and calls run.
Runtime connects components and orders the synchronous loop; workers execute their stages.
"""

from config import RunConfig
from core.runtime.runtime import Runtime


def main(config_path: str):
    config = RunConfig.from_toml(config_path)
    runtime = Runtime(config)
    runtime.run()
