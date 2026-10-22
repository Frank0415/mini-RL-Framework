import json
from pathlib import Path

from .logger import EventLogger


def export_trace(logger: EventLogger, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = [dict(name="process_name", ph="M", pid=1, args=dict(name="RL"))]
    tracks.extend(
        dict(name="thread_name", ph="M", pid=1, tid=tid, args=dict(name=name))
        for name, tid in logger.tracks.items()
    )
    trace = dict(
        traceEvents=tracks + sorted(logger.events, key=lambda event: event["ts"]),
        displayTimeUnit="ms",
        otherData=dict(
            metrics=logger.metrics,
            iteration=logger.iteration,
            total_steps=logger.total_steps,
            finished=logger.finished,
        ),
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
