"""数据管理入口。DataEngine 接收完整采样组，组织经验存储、采样和张量读写，返回 TrainBatch。
它负责 padding、掩码、组编号和版本对齐；算法负责优势值与目标函数，Runtime 负责调用顺序。

Data management entry point. DataEngine receives complete rollout groups and organizes storage, selection, and tensor access to return TrainBatch.
It aligns padding, masks, group IDs, and versions; algorithms compute advantages and objectives, while Runtime orders calls.
"""
