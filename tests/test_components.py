import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from mini_megatron import ModelEngine, Trainer
from mini_sglang import Engine
from mini_sglang.mem_cache.allocator import PagedTokenToKVPoolAllocator
from mini_sglang.mem_cache.memory_pool import ReqToTokenPool
from mini_sglang.mem_cache.radix_cache import RadixCache
from mini_sglang.scheduler.request import Request, RequestStatus
from transformers import LlamaConfig, LlamaForCausalLM

from src.algorithms.grpo import GRPOAlgorithm
from src.core.rollout_worker.worker import RolloutWorker
from src.core.training_worker.worker import TrainingWorker
from src.types import Prompt

from test_grpo_training import make_batch


def make_rollout():
    pool = ReqToTokenPool(4, 128, "cpu")
    allocator = PagedTokenToKVPoolAllocator(4, 32)
    runner = SimpleNamespace(
        device="cpu", runtime=None, req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        radix_cache=RadixCache(4, pool, allocator),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=2)),
        model=SimpleNamespace(load_weights=Mock()),
    )

    def forward(ids, positions, context):
        logits = torch.zeros(len(ids), 16)
        logits[:, 7] = 1
        return logits

    runner.forward = forward
    with patch("mini_sglang.engine.ModelRunner", return_value=runner):
        engine = Engine("unused", device="cpu")
    engine.update_weights([], version=7)
    return engine


class ComponentTests(unittest.TestCase):
    def test_rollout_groups_keep_engine_tokens_logprobs_and_versions(self):
        engine = make_rollout()
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(map(str, ids)))
        worker = RolloutWorker(
            engine, tokenizer, lambda prompt, text: text == prompt.answer,
            max_new_tokens=2,
        )
        prompts = [Prompt("a", [3, 4], "7 7"), Prompt("b", [5, 6], "other")]
        groups = worker.collect(prompts, 2)
        self.assertEqual([group.prompt_id for group in groups], ["a", "b"])
        self.assertNotEqual(groups[0].group_id, groups[1].group_id)
        for group, prompt in zip(groups, prompts):
            self.assertEqual(len(group.samples), 2)
            for sample in group.samples:
                self.assertEqual(sample.group_id, group.group_id)
                self.assertEqual(sample.prompt_token_ids, prompt.prompt_token_ids)
                self.assertEqual(sample.response_token_ids, [7, 7])
                self.assertEqual(len(sample.rollout_log_probs), 2)
                self.assertEqual(sample.behavior_version, 7)
                self.assertEqual(sample.finish_reason, "length")
        self.assertEqual([s.reward for s in groups[0].samples], [1, 1])
        self.assertEqual([s.reward for s in groups[1].samples], [0, 0])
        self.assertEqual(engine.scheduler.requests, {})

    def test_aborted_rollout_is_not_exposed_as_training_data(self):
        request = Request("failed", [3], 2, status=RequestStatus.ABORTED, finish_reason="capacity")
        engine = SimpleNamespace(generate_batch=Mock(return_value=[request]))
        worker = RolloutWorker(engine, Mock(), Mock())
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            worker.collect([Prompt("a", [3], "7")], 1)

    def test_training_worker_uses_mini_trainer_state_and_weight_export(self):
        torch.manual_seed(7)
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        )
        engine = ModelEngine(LlamaForCausalLM(config))
        worker = TrainingWorker(Trainer(engine, learning_rate=0.01), GRPOAlgorithm())
        metrics = worker.update(make_batch())
        self.assertEqual(worker.actor_version, 1)
        self.assertEqual(metrics["actor_version"], 1)
        weights = worker.export_weights()
        self.assertTrue(all(value.device.type == "cpu" for value in weights.values()))
        restored = TrainingWorker(
            Trainer(ModelEngine(LlamaForCausalLM(config))), GRPOAlgorithm(),
        )
        restored.load_state_dict(worker.state_dict())
        self.assertEqual(restored.actor_version, 1)
        for name, value in restored.export_weights().items():
            torch.testing.assert_close(value, weights[name])


if __name__ == "__main__":
    unittest.main()
