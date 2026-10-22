"""策略评估。安装当前训练权重，用固定评估题目采样并计算平均奖励和通过率。
每题采样一个回答，奖励为 1 视为通过；评估恢复本进程的随机状态。

Policy evaluation. Install current training weights and sample fixed evaluation prompts to compute mean reward and pass rate.
Sample one response per prompt, count reward 1 as a pass, and restore this process's random state after evaluation.
"""

import random

import numpy as np
import torch

from ...tasks.prompt_source import SourceBase
from ...observability import EventLogger
from ..rollout_worker.worker import RolloutWorker


class Evaluator:
    def __init__(self, rollout_worker: RolloutWorker, prompts: SourceBase, trace: EventLogger | None = None):
        self.trace = trace or EventLogger()
        self.rollout_worker = rollout_worker
        self.prompts = prompts.next_batch(len(prompts.rows))

    def evaluate(self, weights: dict[str, torch.Tensor], version: int) -> dict[str, float]:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(), self.trace.span("evaluation.run", "evaluation", actor_version=version):
                self.rollout_worker.engine.update_weights(weights.items(), version)
                groups = self.rollout_worker.collect(self.prompts, samples_per_prompt=1)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        rewards = [sample.reward for group in groups for sample in group.samples]
        return {
            "reward_mean": sum(rewards) / len(rewards),
            "pass_rate": sum(reward == 1.0 for reward in rewards) / len(rewards),
        }
