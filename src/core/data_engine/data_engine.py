"""训练数据采样。选择完整回答组，保留样本的分组关系与采样版本。

Training data sampling. Select complete response groups while preserving sample grouping and behavior versions.
"""
import threading

from ...config import RunConfig
from ...types import SampleGroup


class DataEngine:
    def __init__(self, config: RunConfig):
        self.config = config
        self.data_buffer: list[SampleGroup] = []
        self.mutex = threading.Lock()
        self.max_size = 10000000 if not config.data.max_size else config.data.max_size

    def push(self, sample_groups: list[SampleGroup]):
        with self.mutex:
            if self._len() + len(sample_groups) > self.max_size:
                raise ValueError("Data buffer is full")
            self.data_buffer.extend(sample_groups)
    
    def pop(self, n: int) -> list[SampleGroup]:
        with self.mutex:
            ret = self.data_buffer[:n]
            self.data_buffer = self.data_buffer[n:]
            return ret

    def len(self) -> int:
        with self.mutex:
            return self._len()

    def _len(self) -> int:
        return len(self.data_buffer)