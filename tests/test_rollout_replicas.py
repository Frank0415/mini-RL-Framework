from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from src.core.rollout_worker.replica_group import RolloutReplicaGroup
from src.types import GenerationResult


class RolloutReplicaTests(unittest.TestCase):
    def test_requests_run_concurrently_and_return_in_input_order(self):
        barrier = Barrier(2, timeout=5)
        received = []

        def generate(inputs, max_new_tokens):
            received.append(inputs)
            barrier.wait()
            return [GenerationResult(str(ids[0]), ids, [-0.1], 3, "length") for ids in inputs]

        engines = [SimpleNamespace(generate_batch=generate) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            group = RolloutReplicaGroup(engines, executor)
            results = group.generate_batch([[index] for index in range(5)], 1)
        self.assertCountEqual(received, [[[0], [2], [4]], [[1], [3]]])
        self.assertEqual([result.request_id for result in results], list(map(str, range(5))))

    def test_empty_and_small_batches_leave_unused_replicas_idle(self):
        engines = [Mock(), Mock()]
        engines[0].generate_batch.return_value = [GenerationResult("one", [1], [-0.1], 0, "length")]
        with ThreadPoolExecutor(max_workers=2) as executor:
            group = RolloutReplicaGroup(engines, executor)
            self.assertEqual(group.generate_batch([], 1), [])
            self.assertEqual(group.generate_batch([[1]], 1)[0].request_id, "one")
        engines[0].generate_batch.assert_called_once_with([[1]], 1)
        engines[1].generate_batch.assert_not_called()

    def test_weight_iterator_reaches_every_replica_before_return(self):
        engines = [Mock(), Mock()]
        weight = torch.ones(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            group = RolloutReplicaGroup(engines, executor)
            group.update_weights(iter([("weight", weight)]), version=7)
        for engine in engines:
            engine.update_weights.assert_called_once_with((("weight", weight),), 7)


if __name__ == "__main__":
    unittest.main()
