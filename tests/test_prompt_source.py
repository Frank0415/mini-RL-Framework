import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from src.config import DataConfig
from src.tasks.prompt_source import PromptSource


class PromptSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "train.jsonl"
        self.rows = [
            {"id": "a", "question": "first question", "answer": "HIDDEN_A"},
            {"id": "b", "question": "second question", "answer": "HIDDEN_B"},
        ]
        self.path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template.return_value = [7, 8, 9]
        self.source = PromptSource(DataConfig(self.path), self.tokenizer, "Task: {question}")

    def test_formats_only_the_question_and_retains_ground_truth(self):
        prompt = self.source.next_batch(1)[0]
        self.assertEqual(prompt.prompt_id, "a")
        self.assertEqual(prompt.prompt_token_ids, [7, 8, 9])
        self.assertEqual(prompt.answer, "HIDDEN_A")
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Task: first question"}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )

    def test_reads_in_order_and_wraps_within_a_batch(self):
        batch = self.source.next_batch(5)
        self.assertEqual([prompt.prompt_id for prompt in batch], ["a", "b", "a", "b", "a"])
        self.assertEqual(self.source.next_batch(1)[0].prompt_id, "b")

    def test_restored_position_continues_from_the_next_question(self):
        self.source.next_batch(1)
        state = self.source.state_dict()
        self.source.next_batch(1)
        restored = PromptSource(DataConfig(self.path), self.tokenizer, "Task: {question}")
        restored.load_state_dict(state)
        self.assertEqual(restored.next_batch(1)[0].prompt_id, "b")
        self.assertEqual(restored.next_batch(1)[0].prompt_id, "a")


if __name__ == "__main__":
    unittest.main()
