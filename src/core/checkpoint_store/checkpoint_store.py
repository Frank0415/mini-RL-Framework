"""训练状态持久化。每份快照保存为 UTC 时间命名的文件，新快照写完后按数量清理旧文件。
保存模型、优化器、随机状态和运行进度；load 恢复状态并返回进度。

Training-state persistence. Name each snapshot file with its UTC timestamp and prune old files after a successful save.
Save model, optimizer, random state, and run progress; load restores state and returns progress.
"""

from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from ...config import RunConfig
from ...observability import EventLogger


class CheckpointStore:
    def __init__(self, config: RunConfig, trace: EventLogger | None = None):
        self.trace = trace or EventLogger()
        self.checkpoint_count = config.checkpoint.checkpoint_count
        self.checkpoint_location = config.checkpoint.checkpoint_location
        if self.checkpoint_count < 1:
            raise ValueError("checkpoint_count must be at least 1")
        self.checkpoint_location.mkdir(parents=True, exist_ok=True)

    def _checkpoints(self) -> list[Path]:
        return sorted(self.checkpoint_location.glob("checkpoint-????????T??????.??????Z.pt"))

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        random_state: np.random.RandomState | None = None,
        *,
        progress: dict[str, Any] | None = None,
    ) -> Path:
        numpy_rng = np.random if random_state is None else random_state
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": numpy_rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "progress": progress if progress is not None else {},
        }
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = self.checkpoint_location / f"checkpoint-{timestamp}.pt"
        temporary = path.with_suffix(".pt.tmp")
        try:
            with self.trace.span("checkpoint.save", "checkpoint", path=str(path)):
                torch.save(state, temporary)
                temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        for checkpoint in self._checkpoints()[:-self.checkpoint_count]:
            checkpoint.unlink()
        return path

    def load(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        random_state: np.random.RandomState | None = None,
        *,
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        if path is None:
            checkpoints = self._checkpoints()
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints in {self.checkpoint_location}")
            checkpoint = checkpoints[-1]
        else:
            checkpoint = Path(path)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        numpy_rng = np.random if random_state is None else random_state
        numpy_rng.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return state["progress"]
