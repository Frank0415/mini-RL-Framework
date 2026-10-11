"""后续并发训练中选择完整回答组。Sampler 根据训练策略选择数据。
同步版本由 DataEngine 按进入顺序取出本轮回答组。

Select complete response groups for future concurrent training. Sampler selects data according to the training policy.
The synchronous version uses DataEngine to take the current iteration's groups in arrival order.
"""
