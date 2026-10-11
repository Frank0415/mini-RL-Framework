"""RL 执行模块。Runtime 协调 worker、DataEngine、在线权重同步与持久化 checkpoint。
算法定义更新目标，任务定义交互与评分，worker 调用各自的 mini 引擎。

RL execution modules. Runtime coordinates workers, DataEngine, online weight synchronization, and checkpoint storage.
Algorithms define updates, tasks define interaction and scoring, and workers call their mini engines.
"""
