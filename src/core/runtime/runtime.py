"""同步 RL 运行的协调模块。Runtime 连接任务源、worker、DataEngine、checkpoint 与日志。
每轮依次取题、采样、组装 batch、训练、安装权重，安装完成后开始下一轮。
Runtime 管理轮次并触发评估与保存；ResourceScheduler 负责 worker 的资源与生命周期。

Coordinates a synchronous RL run. Runtime connects task sources, workers, DataEngine, checkpoints, and logging.
Each iteration reads tasks, collects responses, builds a batch, trains, and installs weights before the next iteration.
Runtime tracks iterations and triggers evaluation and saving; ResourceScheduler handles worker resources and lifecycle.
"""

from config import RunConfig
from tasks.prompt_source import PromptSource
from transformers import AutoTokenizer

class Runtime:
    def __init__(self, config: RunConfig):
        self.config = config

    def run(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.path,local_files_only=True)
        source = PromptSource(self.config.data, tokenizer, self.config.task.prompt_template)