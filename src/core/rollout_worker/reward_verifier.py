"""任务评分加载。按配置路径加载 verifier 的 score 函数，提供题目标签与回答文本的评分入口。

Task scoring loader. Load the verifier's score function from the configured path and expose scoring for task labels and response text.
"""
from collections.abc import Callable
from runpy import run_path

from ...config import VerifierConfig
from ...types import Prompt


class RewardVerifier:
    def __init__(self, config: VerifierConfig):
        self.verifier_config = config
        self.score: Callable[[Prompt, str], float] = run_path(str(config.path))["score"]
