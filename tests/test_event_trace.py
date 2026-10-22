import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from mini_megatron import ModelEngine, Trainer
from transformers import LlamaConfig, LlamaForCausalLM

from src.algorithms.grpo import GRPOAlgorithm
from src.config import TrainingConfig
from src.core.data_engine.data_engine import DataEngine
from src.core.rollout_worker.worker import RolloutWorker
from src.core.training_worker.worker import TrainingWorker
from src.observability import EventLogger, export_trace
from src.types import GenerationResult, Prompt


class TraceTests(unittest.TestCase):
    def test_samples_link_rollout_to_repeated_training_steps(self):
        torch.manual_seed(7)
        trace = EventLogger()
        trace.iteration = 1
        trace.total_steps = 4

        class Backend:
            def generate_batch(self, inputs, max_new_tokens):
                return [
                    GenerationResult(f"sample-{index}", [7 + index % 2], [-1.0], 0, "eos")
                    for index in range(len(inputs))
                ]

        rollout = RolloutWorker(
            Backend(), SimpleNamespace(decode=lambda ids, **kw: str(ids[0])),
            lambda prompt, text: float(text == prompt.answer), trace=trace,
        )
        groups = rollout.collect([Prompt("a", [1, 2], "7"), Prompt("b", [1, 3], "7")], 2)
        config = SimpleNamespace(
            data=SimpleNamespace(max_size=10), training=TrainingConfig(minibatch_size=2, update_epochs=2),
        )
        data = DataEngine(config, trace=trace)
        data.push(groups)
        batch = data.to_train_batch(data.pop_batch())
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        ))
        worker = TrainingWorker(
            Trainer(ModelEngine(model), learning_rate=.001), GRPOAlgorithm(),
            config.training, trace=trace,
        )
        worker.update(batch)
        self.assertEqual([point["step"] for point in trace.metrics], [1, 2, 3, 4])
        minibatches = [event for event in trace.events if event["name"] == "train.minibatch"]
        self.assertEqual([event["args"]["epoch"] for event in minibatches], [1, 1, 2, 2])
        self.assertTrue(all(len(event["args"]["sample_ids"]) == 2 for event in minibatches))
        for sample_id in batch.sample_ids:
            path = [
                event["name"] for event in trace.events
                if event["ph"] == "i" and event.get("args", {}).get("sample_id") == sample_id
            ]
            self.assertEqual(path[:5], [
                "rollout.enter", "reward.ready", "data.enqueue", "data.dequeue", "batch.ready",
            ])
            self.assertEqual(path.count("train.minibatch.enter"), 2)
        starts = {event["id"] for event in trace.events if event["ph"] == "s"}
        finishes = {event["id"] for event in trace.events if event["ph"] == "f"}
        self.assertEqual(starts, finishes)
        self.assertEqual({event["cat"] for event in trace.events if event["ph"] in ("s", "f")}, {"sample"})
        trace.finished = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            export_trace(trace, path)
            saved = json.loads(path.read_text())
        self.assertEqual(saved["otherData"]["metrics"], trace.metrics)
        self.assertEqual(saved["otherData"]["total_steps"], 4)
        self.assertTrue(saved["otherData"]["finished"])


if __name__ == "__main__":
    unittest.main()
