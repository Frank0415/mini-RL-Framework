from contextlib import contextmanager
from itertools import count
from time import perf_counter_ns


class EventLogger:
    def __init__(self):
        self.started = perf_counter_ns()
        self.events = []
        self.metrics = []
        self.tracks = {}
        self.samples = {}
        self.flow_ids = count(1)
        self.iteration = 0
        self.total_steps = 0
        self.finished = False

    def timestamp(self):
        return (perf_counter_ns() - self.started) / 1000

    def _event(self, name, track, phase, timestamp, **fields):
        tid = self.tracks.setdefault(track, len(self.tracks) + 1)
        event = dict(name=name, cat=track, ph=phase, ts=timestamp, pid=1, tid=tid)
        event.update(fields)
        self.events.append(event)
        return event

    @contextmanager
    def span(self, name, track, **args):
        start = self.timestamp()
        args = dict(iteration=self.iteration, **args)
        yield args
        self._event(name, track, "X", start, dur=self.timestamp() - start, args=args)

    def event(self, name, track, **args):
        self._event(name, track, "i", self.timestamp(), s="t", args=dict(iteration=self.iteration, **args))

    def sample(self, name, sample_id, track, timestamp=None, **args):
        timestamp = self.timestamp() if timestamp is None else timestamp
        point = self._event(name, track, "i", timestamp, s="t", args=dict(
            sample_id=sample_id, iteration=self.iteration, **args,
        ))
        if sample_id in self.samples:
            previous = self.samples[sample_id]
            flow_id = next(self.flow_ids)
            self._event("sample", previous["cat"], "s", previous["ts"], id=flow_id, cat="sample")
            self._event("sample", track, "f", timestamp, id=flow_id, bp="e", cat="sample")
        self.samples[sample_id] = point

    def training(self, step, **values):
        timestamp = self.timestamp()
        self.metrics.append(dict(step=step, iteration=self.iteration, ts=timestamp, **values))
        self._event("training", "progress", "C", timestamp, args=dict(step=step, loss=values["loss"]))
