"""数据管理入口。DataEngine 接收完整回答组，组织经验存储、数据选择和张量读写，返回 TrainBatch。
它负责 padding、掩码、组编号与采样版本对齐；算法负责优势值和损失，Runtime 负责调用顺序。

Data management entry point. DataEngine receives complete response groups and organizes storage, selection, and tensor access to return TrainBatch.
It aligns padding, masks, group IDs, and behavior versions; algorithms compute advantages and loss, while Runtime orders calls.
"""
