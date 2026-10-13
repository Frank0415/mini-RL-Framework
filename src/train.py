"""运行触发入口。读取实验配置并启动运行。

Run entry point. Read experiment configuration and start a run.
"""

import argparse

from .config import RunConfig
from .core.runtime.runtime import Runtime
from .types import SampleGroup


def main(config_path: str):
    config = RunConfig.from_toml(config_path)
    runtime = Runtime(config)
    return runtime.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    for group in main(args.config_path):
        for sample in group.samples:
            print(
                f"prompt={group.prompt_id} version={sample.behavior_version} "
                f"reward={sample.reward} tokens={len(sample.response_token_ids)}"
            )
