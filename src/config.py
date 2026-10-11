"""实验配置。RunConfig 读取 TOML，构造分区 dataclass，并解析数据、模型和 verifier 路径。
train 加载配置，Runtime 将配置交给对应组件。

Experiment configuration. RunConfig reads TOML, builds section dataclasses, and resolves data, model, and verifier paths.
train loads the configuration; Runtime passes it to the corresponding components.
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


@dataclass
class VerifierConfig:
    path: Path


@dataclass
class ModelConfig:
    path: Path


@dataclass
class RolloutConfig:
    samples_per_prompt: int = 4
    max_new_tokens: int = 32


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5
    minibatch_size: int = 4
    update_epochs: int = 1


@dataclass
class RunConfig:
    task: TaskConfig
    data: DataConfig
    verifier: VerifierConfig
    model: ModelConfig
    rollout: RolloutConfig
    training: TrainingConfig

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
        return cls(
            task=TaskConfig(**config["task"]),
            data=DataConfig(**config["data"]),
            verifier=VerifierConfig(**config["verifier"]),
            model=ModelConfig(**config["model"]),
            rollout=RolloutConfig(**config["rollout"]),
            training=TrainingConfig(**config["training"]),
        )
