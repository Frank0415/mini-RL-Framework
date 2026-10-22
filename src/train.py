"""运行触发入口。读取实验配置并启动运行。

Run entry point. Read experiment configuration and start a run.
"""

import sys

from .config import RunConfig
from .core.runtime.runtime import Runtime


def main(config_path: str) -> None:
    config = RunConfig.from_toml(config_path)
    runtime = Runtime(config)
    runtime.run()


if __name__ == "__main__":
    main(sys.argv[1])
