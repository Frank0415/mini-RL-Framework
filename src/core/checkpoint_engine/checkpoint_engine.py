"""在线权重发布、传输与安装。

Online weight publication, transfer, and installation.
"""

from ...config import RunConfig
from .snapshot import Snapshot
from .weight_transport import WeightTransport
from .install_session import InstallSession

from ..training_worker.worker import TrainingWorker
from ..rollout_worker.worker import RolloutWorker

class CheckpointEngine:
    def __init__(self, config: RunConfig):
        self.config = config
        self.snapshot = Snapshot(config)
        self.weight_transport = WeightTransport(config)
        self.install_session = InstallSession(config)
    
    def sync(self, training_worker: TrainingWorker, rollout_worker: RolloutWorker) -> None:
        version = training_worker.actor_version
        weights = training_worker.export_weights()
        rollout_worker.engine.update_weights(weights.items(), version)