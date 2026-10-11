"""为每道题生成多个回答并评分。

RolloutWorker 调用推理引擎生成回答，再调用 RewardVerifier 评分，为结果分配样本编号和组编号。
Trainer 调用 collect，接收完整 SampleGroup。结果中的采样版本来自推理引擎实际安装的权重。

Generate and score multiple responses for each prompt.
RolloutWorker calls the rollout engine, scores responses with RewardVerifier, and assigns sample and group IDs.
Trainer calls collect and receives complete SampleGroup objects. Behavior versions come from the weights actually installed in the rollout engine.

计划接口 / Planned interfaces:
RolloutWorker.collect(prompts, samples_per_prompt)
"""
