"""在线权重快照。Snapshot 从 TrainingWorker 导出本轮参数与版本，供 WeightTransport 传输。
参数副本独立于训练模型，后续更新不会改变已发布的快照。

Online weight snapshot. Snapshot exports the current parameters and version from TrainingWorker for WeightTransport.
Parameter copies remain independent of the training model, so later updates do not change the published snapshot.
"""
