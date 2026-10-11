"""定义模块之间交换的数据。

Prompt 表示题目；Sample 表示一个回答；SampleGroup 保存同题的多个回答；TrainBatch 保存训练张量和样本编号。
数据源、采样模块和训练模块使用同一组数据结构，保留原始词元、概率和实际权重版本。

Define the data exchanged between modules.
Prompt holds a question. Sample holds one response. SampleGroup holds responses to the same question. TrainBatch holds training tensors and sample IDs.
The data source, rollout worker, and training worker share these structures. Keep token IDs, log probabilities, and actual weight versions.

计划接口 / Planned interfaces:
Prompt, Sample, SampleGroup, TrainBatch
"""
