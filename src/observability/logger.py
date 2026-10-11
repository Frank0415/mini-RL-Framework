"""记录训练阶段、事件和数值指标。

RunLogger 记录阶段的开始与结束、权重安装等事件，以及每轮指标，并写入 JSONL 文件。
各模块传入轮次、样本编号和版本。日志器补齐运行编号、进程、线程和时间，供时间线导出使用。

Record training stages, events, and numeric metrics.
RunLogger writes stage start and end records, events such as weight installation, and iteration metrics to JSONL files.
Modules pass iteration, sample, and version IDs. The logger adds run, process, thread, and timestamp fields for trace export.

计划接口 / Planned interfaces:
RunLogger.span(name, **ids), event(name, **fields), metrics(step, values), close()
"""
