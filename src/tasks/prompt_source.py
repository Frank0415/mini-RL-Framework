"""题目数据源。PromptSource 读取数据，应用任务模板和 tokenizer，返回 Prompt。
Runtime 调用 next_batch 取题；标准答案留作评分标签，state_dict 保存读取位置。

Prompt data source. PromptSource reads records, applies the task template and tokenizer, and returns Prompt objects.
Runtime requests batches; answers remain scoring labels, and state_dict preserves the reading position.
"""

import json

from ..config import DataConfig
from ..types import Prompt


class PromptSource:
    def __init__(self, conf_train: DataConfig, tokenizer, task_prompt_template: str):
        with conf_train.train.open(encoding="utf-8") as file:
            self.rows = [json.loads(line) for line in file]
        self.tokenizer = tokenizer
        self.prompt_template = task_prompt_template
        self.position = 0

    def next_batch(self, n: int) -> list[Prompt]:
        prompts = []
        for _ in range(n):
            row = self.rows[self.position]
            text = self.prompt_template.format(question=row["question"])
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            prompts.append(Prompt(row["id"], token_ids, row["answer"]))
            self.position = (self.position + 1) % len(self.rows)
        return prompts

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = state["position"]
