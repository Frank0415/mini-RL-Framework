import os
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from transformers import AutoTokenizer

from mini_megatron import ModelEngine, Trainer
from mini_sglang import Engine
from src.algorithms.grpo import GRPOAlgorithm
from src.types import Prompt, TrainBatch
from src.core.training_worker.worker import TrainingWorker
from src.core.rollout_worker.worker import RolloutWorker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT.parent / (
    "mini-sglang/models/smollm2-135m-instruct/"
    "models--HuggingFaceTB--SmolLM2-135M-Instruct/"
    "snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac"
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GRPOGPUChecks(unittest.TestCase):
    def test_rollout_train_install_and_next_generation(self):
        model_dir = Path(os.environ.get("MINI_RL_MODEL_DIR", DEFAULT_MODEL))
        if not (model_dir / "config.json").is_file():
            self.skipTest("Set MINI_RL_MODEL_DIR to a local model directory")
        torch.manual_seed(7)
        rollout = Engine(model_dir, do_sample=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Calculate 7 + 5. Give a short answer."}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        task = Prompt("question", prompt, "12")
        rollout_worker = RolloutWorker(
            rollout, tokenizer, Mock(side_effect=[1.0, 0.0, 1.0, 0.0, 1.0]),
            max_new_tokens=8,
        )
        groups = rollout_worker.collect([task], 4)
        responses = groups[0].samples
        actor = ModelEngine.from_pretrained(model_dir)
        trainer = Trainer(actor, learning_rate=1e-3)
        worker = TrainingWorker(trainer, GRPOAlgorithm(update_epochs=2))
        size = len(responses)
        response_length = max(len(item.response_token_ids) for item in responses)
        ids = torch.zeros(size, len(prompt) + response_length, dtype=torch.long)
        attention = torch.zeros_like(ids)
        mask = torch.zeros(size, response_length, dtype=torch.bool)
        rollout_log_probs = torch.zeros(size, response_length)
        for index, item in enumerate(responses):
            sequence = prompt + item.response_token_ids
            ids[index, :len(sequence)] = torch.tensor(sequence)
            attention[index, :len(sequence)] = 1
            mask[index, :len(item.response_token_ids)] = True
            rollout_log_probs[index, :len(item.response_token_ids)] = torch.tensor(item.rollout_log_probs)
        batch = TrainBatch(
            input_ids=ids, attention_mask=attention, response_mask=mask,
            prompt_lengths=torch.full((size,), len(prompt)),
            response_lengths=mask.sum(-1), rollout_log_probs=rollout_log_probs,
            rewards=torch.tensor([item.reward for item in responses]),
            behavior_versions=torch.tensor([item.behavior_version for item in responses]),
            sample_ids=[item.sample_id for item in responses], group_ids=[item.group_id for item in responses],
        )
        parameter = actor.model.model.norm.weight.detach().clone()
        metrics = worker.update(batch)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertFalse(torch.equal(parameter, actor.model.model.norm.weight))
        self.assertEqual(worker.actor_version, 2)
        rollout.update_weights(worker.export_weights().items(), worker.actor_version)
        self.assertEqual(rollout.runner.radix_cache.root_node.children, {})
        actual = dict(rollout.runner.model.named_parameters(remove_duplicate=False))
        for name, weight in actor.model.state_dict().items():
            torch.testing.assert_close(actual[name], weight.to(actual[name].dtype), rtol=0, atol=0)
        rollout_worker.max_new_tokens = 4
        next_response = rollout_worker.collect([task], 1)[0].samples[0]
        self.assertEqual(next_response.behavior_version, 2)
        self.assertEqual(len(next_response.rollout_log_probs), len(next_response.response_token_ids))
        self.assertEqual(rollout.scheduler.requests, {})
        print(f"GPU GRPO: actor_version={worker.actor_version}, loss={metrics['loss']:.6f}")


if __name__ == "__main__":
    unittest.main()
