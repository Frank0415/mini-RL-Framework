"""整次 RL 运行的协调模块。Runtime 连接任务源、worker、DataEngine、checkpoint 与日志。
同步阶段按取题、采样、组装 batch、训练、权重安装的顺序执行；安装完成后开始下一轮。
Runtime 管理迭代并触发评估与保存，将资源放置和生命周期交给 ResourceScheduler。

Coordinates an RL run. Runtime connects task sources, workers, DataEngine, checkpoints, and logging.
The synchronous schedule reads tasks, collects rollouts, builds a batch, trains, and installs weights before the next iteration.
Runtime tracks iterations and triggers evaluation and saving; ResourceScheduler handles placement and lifecycle.
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