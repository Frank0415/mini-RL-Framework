"""采样 worker。连接任务交互、推理与评分，返回带实际采样版本的完整回答组。
mini-sglang 执行模型推理，DataEngine 接收生成并评分的经验。

Rollout workers connect task interaction, inference, and scoring, returning complete response groups with actual behavior versions.
mini-sglang executes model inference; DataEngine receives generated and scored experience.
"""
