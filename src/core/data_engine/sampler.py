"""训练数据选择。Sampler 从完成经验中选择完整回答组，保留 GRPO 的同题组关系。
DataEngine 调用它选择下一批数据；同步首版可以按进入顺序取组。

Training data selection. Sampler chooses complete response groups from finished experience, preserving GRPO task grouping.
DataEngine calls it to select the next batch; the synchronous version can use arrival order.
"""
