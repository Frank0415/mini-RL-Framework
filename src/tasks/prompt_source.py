"""读取题目并生成模型输入。

PromptSource 读取 JSONL 题目，应用输入模板并分词，保存数据读取位置。
Trainer 调用 next_batch 取题。采样和训练共用这里生成的词元编号；恢复训练时加载读取状态。

Read prompts and prepare model inputs.
PromptSource reads JSONL records, applies the input template, tokenizes prompts, and tracks the reading position.
Trainer calls next_batch. Rollout and training use the same token IDs. Restore the reading state when resuming training.

计划接口 / Planned interfaces:
PromptSource.next_batch(n), state_dict(), load_state_dict(state)
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
