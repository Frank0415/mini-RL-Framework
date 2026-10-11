"""实验配置。RunConfig 把 TOML 转为分区 dataclass，并解析数据、模型和 verifier 路径。
入口读取配置，Runtime 将各部分交给负责的模块。

Experiment configuration. RunConfig converts TOML into section dataclasses and resolves resource paths.
The entry point loads the configuration; Runtime passes each section to its owning module.
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
