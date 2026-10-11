"""时间线导出。将 RunLogger 的阶段事件和指标转为 Perfetto 可读取的 Chrome Trace JSON。
展示阶段区间、瞬时事件与计数；CUDA kernel 时间使用 GPU profiler 的记录。

Timeline export. Convert RunLogger stage events and metrics into Chrome Trace JSON for Perfetto.
Show stage intervals, instant events, and counters; GPU profiler records supply CUDA kernel timing.
"""
