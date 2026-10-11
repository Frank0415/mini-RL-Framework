"""训练 worker。连接算法与 mini-megatron，提供训练、权重导出和状态恢复。
评估模块执行评估任务；Runtime 决定训练、评估和保存的时机。

Training workers connect algorithms to mini-megatron and expose training, weight export, and state recovery.
The evaluation module executes evaluation tasks; Runtime schedules training, evaluation, and saving.
"""
