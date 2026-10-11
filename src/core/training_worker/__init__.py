"""训练 worker。管理算法需要的模型角色，调用 ModelEngine 更新参数，并提供评估与状态接口。
Runtime 负责何时训练、评估和保存。

Training workers manage the model roles required by the algorithm, call ModelEngine, and expose evaluation and state operations.
Runtime decides when to train, evaluate, and save.
"""
