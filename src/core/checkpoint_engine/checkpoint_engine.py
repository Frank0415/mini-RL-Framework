"""在线权重发布、传输与安装。

Online weight publication, transfer, and installation.
"""

from ...config import RunConfig
from ...observability import EventLogger
from .snapshot import Snapshot
from .weight_transport import WeightTransport
from .install_session import InstallSession

from ..training_worker.worker import TrainingWorker
from ..rollout_worker.worker import RolloutWorker

class CheckpointEngine:
    def __init__(self, config: RunConfig, trace: EventLogger | None = None):
        self.config = config
        self.trace = trace or EventLogger()
        self.snapshot = Snapshot(config)
        self.weight_transport = WeightTransport(config)
        self.install_session = InstallSession(config)
    
    def sync(self, training_worker: TrainingWorker, rollout_worker: RolloutWorker) -> None:
        version = training_worker.actor_version
        with self.trace.span("weights.export", "weights", actor_version=version):
            weights = training_worker.export_weights()
        with self.trace.span("weights.install", "weights", actor_version=version):
            rollout_worker.engine.update_weights(weights.items(), version)
