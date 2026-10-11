"""在线权重快照。Snapshot 冻结本轮参数，携带模型标识与版本，供 WeightTransport 传输。
快照使用独立参数副本，后续优化器更新不会改变已发布版本。

Online weight snapshot. Snapshot freezes parameters with model identity and version for WeightTransport.
Independent parameter copies keep later optimizer updates from changing the published version.
"""
