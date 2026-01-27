"""
Efficient Math Environment with self-improving efficiency reward.

The reward function incentivizes correct AND short answers:
- If incorrect: reward = 0 (no gaming with short gibberish)
- If correct: reward = best_so_far / num_tokens
  - First correct answer sets the baseline (reward = 1.0)
  - Shorter than best: reward > 1.0 (strong incentive)
  - Longer than best: reward < 1.0 (still positive, but penalized)

The target gets harder as the model finds shorter solutions (self-improving).
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Literal, Sequence, cast

import chz
import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_env import (
    MathEnv,
    extract_gsm8k_final_answer,
    safe_grade,
)
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree


class EfficientMathEnv(MathEnv):
    """MathEnv with self-improving efficiency reward.

    Tracks the best (shortest) correct answer seen for each problem globally.
    Rewards correct answers proportionally to how much shorter they are
    compared to the best seen so far.
    """

    # Class-level global tracking of best lengths per problem
    # Key: problem hash, Value: best (shortest) token count seen
    best_lengths: ClassVar[dict[str, int]] = {}

    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        problem_id: str | None = None,
    ):
        super().__init__(problem, answer, renderer, convo_prefix, grader, timeout)
        # Use provided problem_id or hash of problem text
        self.problem_id = problem_id or str(hash(problem))

    @classmethod
    def question_suffix(cls) -> str:
        """Add instruction for boxed format."""
        return " Provide a numerical answer without units, written inside \\boxed{}."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    async def step(self, action: Action) -> StepResult:
        """Execute action and compute efficiency reward.

        Reward formula:
        - If incorrect: reward = 0
        - If correct: reward = best_so_far / num_tokens
        """
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        # Check format and answer
        correct_format = float(parse_success) and float(self.check_format(content))
        correct_answer = self.check_answer(content)

        # Count tokens in the response
        num_tokens = len(action)

        # Compute efficiency reward
        if correct_answer:
            # Get current best for this problem
            current_best = self.best_lengths.get(self.problem_id, num_tokens)

            # Compute reward: ratio of best to current
            reward = current_best / num_tokens

            # Update global best if this is shorter
            self.best_lengths[self.problem_id] = min(current_best, num_tokens)

            # Log the improvement
            if num_tokens < current_best:
                logtree.log_text(
                    f"New best for problem {self.problem_id[:8]}: "
                    f"{current_best} -> {num_tokens} tokens (improvement: {current_best - num_tokens})"
                )
        else:
            # Incorrect answer gets 0 reward
            reward = 0.0

        # Log the attempt
        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Correct: {'✓' if correct_answer else '✗'}, "
            f"Tokens: {num_tokens}, Reward: {reward:.3f}"
        )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": float(correct_answer),
                "num_tokens": num_tokens,
                "efficiency_reward": reward,
                "best_tokens": self.best_lengths.get(self.problem_id, num_tokens),
            },
        )

    @classmethod
    def reset_best_lengths(cls):
        """Reset the global best lengths tracking. Useful for testing."""
        cls.best_lengths = {}

    @classmethod
    def get_best_lengths_summary(cls) -> dict[str, int]:
        """Get a summary of current best lengths."""
        return dict(cls.best_lengths)


@dataclass(frozen=True)
class EfficientProblemGroupBuilder(ProblemGroupBuilder):
    """Builder for groups of EfficientMathEnv instances."""

    env_thunk: partial
    num_envs: int
    dataset_name: str = "gsm8k_efficient"

    async def make_envs(self) -> Sequence[EfficientMathEnv]:
        return [self.env_thunk() for _ in range(self.num_envs)]


def get_fixed_gsm8k_problems(num_problems: int = 100, seed: int = 42) -> Dataset:
    """Load a fixed set of GSM-8K problems for consistent training/evaluation."""
    ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split="train"))
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(num_problems, len(ds))))


class EfficientGsm8kDataset(RLDataset):
    """GSM-8K dataset with efficient math environments."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        num_problems: int = 100,
        seed: int = 42,
        n_epochs: int = 1,
    ):
        self.ds = get_fixed_gsm8k_problems(num_problems, seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.n_epochs = n_epochs
        self._batches_per_epoch = math.ceil(len(self.ds) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        # Map index to epoch and batch within epoch
        batch_within_epoch = index % self._batches_per_epoch
        batch_start = batch_within_epoch * self.batch_size
        batch_end = min((batch_within_epoch + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None
        ]

    def __len__(self) -> int:
        return self._batches_per_epoch * self.n_epochs

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> EfficientProblemGroupBuilder | None:
        try:
            problem = x["question"]
            answer = extract_gsm8k_final_answer(x["answer"])
            # Use a stable problem ID based on the problem text
            problem_id = str(hash(problem))
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None
        return EfficientProblemGroupBuilder(
            env_thunk=partial(
                EfficientMathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                problem_id=problem_id,
            ),
            num_envs=group_size,
        )


@chz.chz
class EfficientGsm8kDatasetBuilder(RLDatasetBuilder):
    """Builder for efficient GSM-8K dataset."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    num_problems: int = 10
    seed: int = 42
    n_epochs: int = 1
    convo_prefix: list[renderers.Message] | None = None

    async def __call__(self) -> tuple[EfficientGsm8kDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            EfficientGsm8kDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=self.convo_prefix,
                num_problems=self.num_problems,
                seed=self.seed,
                n_epochs=self.n_epochs,
            ),
            None,  # No separate test dataset
        )
