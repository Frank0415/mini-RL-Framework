"""读取题目并生成模型输入。

PromptSource 读取 JSONL 题目，应用输入模板并分词，保存数据读取位置。
Trainer 调用 next_batch 取题。采样和训练共用这里生成的词元编号；恢复训练时加载读取状态。

Read prompts and prepare model inputs.
PromptSource reads JSONL records, applies the input template, tokenizes prompts, and tracks the reading position.
Trainer calls next_batch. Rollout and training use the same token IDs. Restore the reading state when resuming training.

计划接口 / Planned interfaces:
PromptSource.next_batch(n), state_dict(), load_state_dict(state)
"""
