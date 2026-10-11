"""训练数据选择。Sampler 从完成经验中选择下一批完整回答组，保留 GRPO 的组关系。
由 DataEngine 调用，初版可以按进入顺序取组。

Training data selection. Sampler chooses complete response groups from finished experience and preserves GRPO grouping.
DataEngine calls it; the initial implementation can use arrival order.
"""
