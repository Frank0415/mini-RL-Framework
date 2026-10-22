"""题目数据源。SourceBase 负责取题、模板、tokenizer 和读取位置。
PromptSource 读取 data.train，EvalSource 读取 data.eval；标准答案保留为评分标签。

Prompt data sources. SourceBase handles prompt batches, templates, tokenization, and reading position.
PromptSource reads data.train and EvalSource reads data.eval; answers remain scoring labels.
"""

import json
from typing import Any

from ..config import DataConfig
from ..types import Prompt


class SourceBase:
    def __init__(self, rows: list[dict[str, Any]], tokenizer, prompt_template: str, batch_size: int = 32):
        self.rows = rows
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.position = 0
        self.batch_size = batch_size

    def next_batch(self, n: int) -> list[Prompt]:
        prompts = []
        for _ in range(n):
            if self.is_finished():
                break
            row = self.rows[self.position]
            text = self.prompt_template.format(question=row["question"])
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            prompts.append(Prompt(row["id"], token_ids, row["answer"]))
            self.position += 1
        return prompts

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def is_finished(self) -> bool:
        return self.position >= len(self.rows)

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = state["position"]


class PromptSource(SourceBase):
    def __init__(self, conf_train: DataConfig, tokenizer, task_prompt_template: str):
        with conf_train.train.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file]
        super().__init__(rows, tokenizer, task_prompt_template, conf_train.batch_size)


class EvalSource(SourceBase):
    def __init__(self, config: DataConfig, tokenizer, task_prompt_template: str):
        if config.eval is None:
            raise ValueError("data.eval is required for evaluation")
        with config.eval.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file]
        super().__init__(rows, tokenizer, task_prompt_template, config.batch_size)
