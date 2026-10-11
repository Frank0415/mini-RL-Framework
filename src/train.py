"""连接模块并启动同步 RL 训练。

读取配置，创建分词器、训练模型、推理模型和优化器，再创建 Trainer。退出时关闭日志。
main 负责创建对象；Trainer 负责执行训练循环。

Build the modules and start synchronous RL training.
Read the configuration. Create the tokenizer, training model, rollout model, optimizer, and Trainer. Close the logger on exit.
main builds the objects. Trainer runs the training loop.

计划接口 / Planned interfaces:
main(config_path), build(config)
"""
