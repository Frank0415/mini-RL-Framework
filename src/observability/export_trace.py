"""把阶段日志导出为 Perfetto 时间线。

export_trace 读取事件和指标，生成 Chrome Trace JSON，展示阶段区间、瞬时事件和数值曲线。
训练结束或暂停后运行导出入口。Python 阶段耗时表示调用区间；GPU 执行时间由独立性能分析记录。

Export stage logs as a Perfetto timeline.
export_trace reads events and metrics, then writes Chrome Trace JSON with stage intervals, instant events, and counters.
Run the exporter after training ends or pauses. Python spans measure call intervals. Separate profiling records GPU execution time.

计划接口 / Planned interfaces:
export_trace(run_dir, output), main()
"""
