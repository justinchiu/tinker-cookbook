"""
Efficient Math Environment with length-penalized reward.

The reward function incentivizes correct AND short answers:
- If incorrect: reward = 0 (no gaming with short gibberish)
- If correct: reward = max_tokens - num_tokens (shorter = higher reward)

Correct answers always get positive reward, with shorter solutions
getting higher rewards. Incorrect answers get 0.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence, cast

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
    StrategyId,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree


class EfficientMathEnv(MathEnv):
    """MathEnv with length-penalized reward.

    Rewards correct answers with (max_tokens - num_tokens), so shorter correct
    solutions get higher positive rewards. Incorrect answers get 0.
    """

    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        max_tokens: int = 4096,
        question_suffix_extra: str | None = None,
    ):
        super().__init__(problem, answer, renderer, convo_prefix, grader, timeout)
        self.max_tokens = max_tokens
        self.question_suffix_extra = question_suffix_extra

    @classmethod
    def question_suffix(cls) -> str:
        """Add instruction for boxed format."""
        return " Provide a numerical answer without units, written inside \\boxed{}."

    def get_question(self) -> str:
        suffix = self.question_suffix()
        if self.question_suffix_extra:
            return self.problem + suffix + self.question_suffix_extra
        return self.problem + suffix

    async def step(self, action: Action) -> StepResult:
        """Execute action and compute efficiency reward.

        Reward formula:
        - If incorrect: reward = 0
        - If correct: reward = max_tokens - num_tokens (shorter = higher)
        """
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        # Check format and answer
        correct_format = float(parse_success) and float(self.check_format(content))
        correct_answer = self.check_answer(content)

        # Count tokens in the response
        num_tokens = len(action)

        # Compute efficiency reward: correct * (max_tokens - num_tokens)
        # Correct answers get positive reward, shorter = higher
        # Incorrect answers get 0
        if correct_answer:
            reward = float(self.max_tokens - num_tokens)
        else:
            reward = 0.0

        # Log the attempt
        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Correct: {'✓' if correct_answer else '✗'}, "
            f"Tokens: {num_tokens}, Reward: {reward:.1f}"
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
            },
        )


class ExItStrategy(StrategyId):
    IID = "iid"
    PROMPT_AUG = "prompt_aug"
    ANSWER_HINT = "answer_hint"


@dataclass(frozen=True)
class ExItStrategyConfig:
    strategy_id: ExItStrategy
    sampling_prefix: list[renderers.Message]
    training_prefix: list[renderers.Message]


ANSWER_HINT_TEXT = (
    "The answer is {answer}. Solve this problem as if you didn't know the answer, "
    "but then stop once you reach the answer. No need to verify or try again."
)



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
        strategy_configs: list[ExItStrategyConfig] | None = None,
        num_problems: int = 100,
        seed: int = 42,
        n_epochs: int = 1,
        max_tokens: int = 4096,
    ):
        self.ds = get_fixed_gsm8k_problems(num_problems, seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        base_prefix = convo_prefix or []
        self.strategy_configs = strategy_configs or [
            ExItStrategyConfig(
                strategy_id=ExItStrategy.IID,
                sampling_prefix=base_prefix,
                training_prefix=base_prefix,
            )
        ]
        self.n_epochs = n_epochs
        self.max_tokens = max_tokens
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
            for builder in self._make_env_group_builders(row, self.group_size)
        ]

    def __len__(self) -> int:
        return self._batches_per_epoch * self.n_epochs

    def _make_env_group_builders(
        self, x: dict[str, str], group_size: int
    ) -> list[EfficientProblemGroupBuilder]:
        try:
            problem = x["question"]
            answer = extract_gsm8k_final_answer(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return []
        question = problem + EfficientMathEnv.question_suffix()
        builders: list[EfficientProblemGroupBuilder] = []
        for config in self.strategy_configs:
            sampling_prefix = list(config.sampling_prefix)
            training_prefix = list(config.training_prefix)
            question_suffix_extra = None
            if config.strategy_id == ExItStrategy.ANSWER_HINT:
                question_suffix_extra = "\n\n" + ANSWER_HINT_TEXT.format(answer=answer)
            context_transform = None
            if training_prefix != sampling_prefix or question_suffix_extra is not None:
                renderer = self.renderer

                def _transform(
                    _ob: tinker.ModelInput,
                    _turn_idx: int,
                    *,
                    _question: str = question,
                    _renderer: renderers.Renderer = renderer,
                    _training_prefix: list[renderers.Message] = training_prefix,
                ) -> tinker.ModelInput:
                    convo = _training_prefix + [{"role": "user", "content": _question}]
                    return _renderer.build_generation_prompt(convo)

                context_transform = _transform
            builders.append(
                EfficientProblemGroupBuilder(
                    env_thunk=partial(
                        EfficientMathEnv,
                        problem,
                        answer,
                        self.renderer,
                        convo_prefix=sampling_prefix,
                        max_tokens=self.max_tokens,
                        question_suffix_extra=question_suffix_extra,
                    ),
                    num_envs=group_size,
                    strategy_id=config.strategy_id,
                    context_transform=context_transform,
                )
            )
        return builders


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
    max_tokens: int = 4096
    convo_prefix: list[renderers.Message] | None = None
    strategy_configs: list[ExItStrategyConfig] | None = None

    async def __call__(self) -> tuple[EfficientGsm8kDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            EfficientGsm8kDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=self.convo_prefix,
                strategy_configs=self.strategy_configs,
                num_problems=self.num_problems,
                seed=self.seed,
                n_epochs=self.n_epochs,
                max_tokens=self.max_tokens,
            ),
            None,  # No separate test dataset
        )
