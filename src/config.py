"""实验配置。读取 TOML，构造分区 dataclass，并解析数据、模型和 verifier 路径。

Experiment configuration. Read TOML, build section dataclasses, and resolve data, model, and verifier paths.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class TaskConfig:
    prompt_template: str


@dataclass
class DataConfig:
    train: Path
    max_size: int = 10000000
    batch_size: int = 32
    eval: Path | None = None


@dataclass
class VerifierConfig:
    path: Path


@dataclass
class ModelConfig:
    path: Path


@dataclass
class RolloutConfig:
    engine: str = "mini-sglang"
    samples_per_prompt: int = 4
    samples_per_update: int = 32
    max_new_tokens: int = 128


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5
    minibatch_size: int = 4
    update_epochs: int = 1
    eval_every: int = 0

@dataclass
class CheckpointConfig:
    checkpoint_count: int = 3
    checkpoint_location: Path = Path("checkpoint")


@dataclass
class RunConfig:
    task: TaskConfig
    data: DataConfig
    verifier: VerifierConfig
    model: ModelConfig
    rollout: RolloutConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        if path.is_dir():
            path = path / "config.toml"
        path = path.resolve()
        with path.open("rb") as file:
            config = tomllib.load(file)
        for section, key in (
            ("data", "train"), ("verifier", "path"), ("model", "path"),
        ):
            config[section][key] = (path.parent / config[section][key]).resolve()
        if config["data"].get("eval") is not None:
            config["data"]["eval"] = (path.parent / config["data"]["eval"]).resolve()
        checkpoint = config.setdefault("checkpoint", {})
        checkpoint["checkpoint_location"] = (
            path.parent / checkpoint.get("checkpoint_location", "checkpoint")
        ).resolve()
        return cls(
            task=TaskConfig(**config["task"]),
            data=DataConfig(**config["data"]),
            verifier=VerifierConfig(**config["verifier"]),
            model=ModelConfig(**config["model"]),
            rollout=RolloutConfig(**config["rollout"]),
            training=TrainingConfig(**config["training"]),
            checkpoint=CheckpointConfig(**checkpoint),
        )
