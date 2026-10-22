"""运行协调入口。连接运行组件，按数据源批次采样并通过数据缓冲传递完整回答组。
取出回答组后转换为 TrainBatch，交给 TrainingWorker 更新参数。

Run coordination entry point. Connect components, collect source batches, and pass complete response groups through the data buffer.
Convert consumed response groups into TrainBatch and pass it to TrainingWorker for parameter updates.
"""

from datetime import datetime, timezone

from mini_megatron import ModelEngine, Trainer
from transformers import AutoTokenizer

from ...algorithms.grpo import GRPOAlgorithm
from ...config import RunConfig
from ...observability import EventLogger, export_trace
from ...tasks.prompt_source import EvalSource, PromptSource
from ..data_engine.data_engine import DataEngine
from ..checkpoint_engine.checkpoint_engine import CheckpointEngine
from ..checkpoint_store.checkpoint_store import CheckpointStore
from ..rollout_worker.backends import build_rollout_engine
from ..rollout_worker.reward_verifier import RewardVerifier
from ..rollout_worker.worker import RolloutWorker
from ..training_worker.evaluation import Evaluator
from ..training_worker.worker import TrainingWorker


class Runtime:
    def __init__(self, config: RunConfig):
        self.config = config
        self.trace = EventLogger()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.trace_path = config.trace / run_id / "trace.json"

    def run(self) -> None:
        self.trace.event("run.start", "runtime", model=str(self.config.model.path))
        with self.trace.span("tokenizer.load", "runtime"):
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.path, local_files_only=True
            )
        source = PromptSource(
            self.config.data, tokenizer, self.config.task.prompt_template
        )
        verifier = RewardVerifier(self.config.verifier)
        with self.trace.span("rollout.load", "runtime"):
            rollout_engine = build_rollout_engine(self.config.model, self.config.rollout)
        data_engine = DataEngine(self.config, trace=self.trace)
        rollout_worker = RolloutWorker(
            rollout_engine,
            tokenizer,
            verifier.score,
            max_new_tokens=self.config.rollout.max_new_tokens,
            trace=self.trace,
        )

        evaluator = None
        if self.config.training.eval_every > 0:
            eval_source = EvalSource(
                self.config.data, tokenizer, self.config.task.prompt_template
            )
            evaluator = Evaluator(rollout_worker, eval_source, trace=self.trace)

        with self.trace.span("training.load", "runtime"):
            training_worker = TrainingWorker(
                Trainer(
                    ModelEngine.from_pretrained(self.config.model.path),
                    learning_rate=self.config.training.learning_rate,
                ),
                GRPOAlgorithm(),
                self.config.training,
                evaluator=evaluator,
                trace=self.trace,
            )
        checkpoint_engine = CheckpointEngine(self.config, trace=self.trace)
        checkpoint_store = CheckpointStore(self.config, trace=self.trace)
        prompts_per_update = self.config.rollout.samples_per_update
        group_size = self.config.rollout.samples_per_prompt
        minibatch_size = self.config.training.minibatch_size
        self.trace.total_steps = sum(
            (min(prompts_per_update, len(source.rows) - offset) * group_size + minibatch_size - 1)
            // minibatch_size * self.config.training.update_epochs
            for offset in range(0, len(source.rows), prompts_per_update)
        )
        checkpoint_store.save(
            training_worker.trainer.engine.model,
            training_worker.trainer.optimizer,
            progress={
                "optimizer_step": training_worker.actor_version,
                "batches_trained": training_worker.batches_trained,
                "source": source.state_dict(),
            },
        )
        export_trace(self.trace, self.trace_path)
        print(f"Trace: {self.trace_path}", flush=True)

        while not source.is_finished():
            self.trace.iteration += 1
            self.trace.event("iteration.start", "runtime")
            samples_per_update = self.config.rollout.samples_per_update
            collected_samples = 0
            while not source.is_finished() and collected_samples < samples_per_update:
                try:
                    result = rollout_worker.collect(
                        source.next_batch(1),
                        self.config.rollout.samples_per_prompt,
                    )
                    data_engine.push(result)
                except ValueError as error:
                    print(f"Error: {error}")
                    break
                collected_samples += 1

            export_trace(self.trace, self.trace_path)
            while data_engine.len() > 0:
                sample_groups = data_engine.pop_batch()
                batch = data_engine.to_train_batch(sample_groups)
                with self.trace.span("train.update", "runtime", sample_ids=batch.sample_ids):
                    training_worker.update(batch)
                export_trace(self.trace, self.trace_path)

            checkpoint_engine.sync(training_worker, rollout_worker)
            checkpoint_store.save(
                training_worker.trainer.engine.model,
                training_worker.trainer.optimizer,
                progress={
                    "optimizer_step": training_worker.actor_version,
                    "batches_trained": training_worker.batches_trained,
                    "source": source.state_dict(),
                },
            )
            self.trace.event("iteration.end", "runtime", actor_version=training_worker.actor_version)
            export_trace(self.trace, self.trace_path)

        self.trace.finished = True
        self.trace.event("run.end", "runtime", actor_version=training_worker.actor_version)
        export_trace(self.trace, self.trace_path)
