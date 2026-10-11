"""算法模块。负责优势值、目标函数和本批数据的更新安排。
Runtime 决定全局阶段顺序，mini-megatron 执行模型训练。

Algorithms compute advantages, define objectives, and arrange updates within a batch.
Runtime orders the overall stages; mini-megatron executes model training.
"""
