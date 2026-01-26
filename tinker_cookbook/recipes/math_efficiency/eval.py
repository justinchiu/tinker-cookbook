"""
Evaluation script for math efficiency training.

Measures accuracy, token counts, thinking tokens, and efficiency score.
Uses 100 fixed problems from GSM-8K train set with 4 samples per problem.
"""

import asyncio
import json
import logging
import re
from typing import cast

from tqdm.asyncio import tqdm as async_tqdm

import chz
import tinker
import wandb
from datasets import Dataset, load_dataset
from pydantic import BaseModel
from tqdm import tqdm

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


# Pydantic models for results
class SampleResult(BaseModel):
    """Result for a single sample (one completion for one problem)."""

    response: str
    predicted_answer: str | None
    correct: bool
    total_tokens: int
    thinking_tokens: int


class ProblemResult(BaseModel):
    """Result for a single problem (multiple samples)."""

    problem_id: int
    problem: str
    reference_answer: str
    samples: list[SampleResult]  # 4 samples per problem
    pass_rate: float  # fraction correct out of samples
    mean_tokens: float  # average tokens across samples
    mean_thinking_tokens: float


class EvalResults(BaseModel):
    """Full evaluation results."""

    model_name: str
    checkpoint_path: str | None
    num_problems: int
    samples_per_problem: int
    problems: list[ProblemResult]

    # Aggregate metrics
    accuracy: float  # mean pass_rate across problems
    mean_tokens: float
    median_tokens: float
    p90_tokens: float
    mean_thinking_tokens: float
    efficiency: float  # accuracy / (mean_tokens / 100)


def get_fixed_gsm8k_problems(num_problems: int = 100, seed: int = 42) -> Dataset:
    """Load a fixed set of GSM-8K problems for consistent evaluation."""
    ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split="train"))
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(num_problems, len(ds))))


def count_thinking_tokens(response: str, tokenizer) -> int:
    """Count the number of tokens inside <think>...</think> blocks."""
    # Find all thinking blocks
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return 0
    # Tokenize and count
    total_thinking_tokens = 0
    for match in matches:
        tokens = tokenizer.encode(match, add_special_tokens=False)
        total_thinking_tokens += len(tokens)
    return total_thinking_tokens


def extract_answer_from_response(response: str) -> str | None:
    """Extract the answer from a model response."""
    try:
        return extract_boxed(response)
    except ValueError:
        return None


def grade_response(response: str, reference_answer: str, timeout: float = 1.0) -> bool:
    """Grade a response against the reference answer."""
    predicted = extract_answer_from_response(response)
    if predicted is None:
        return False
    try:
        return grade_answer(predicted, reference_answer)
    except Exception:
        return False


async def sample_completions(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    problem: str,
    num_samples: int,
    max_tokens: int,
    temperature: float,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> list[tuple[str, list[int]]]:
    """Sample multiple completions for a problem with timeout and retry."""
    # Build the prompt
    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    messages = [{"role": "user", "content": problem + question_suffix}]
    model_input = renderer.build_generation_prompt(messages)
    stop_condition = renderer.get_stop_sequences()

    # Sample with timeout and retry
    for attempt in range(max_retries):
        try:
            result = await asyncio.wait_for(
                sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=num_samples,
                    sampling_params=tinker.SamplingParams(
                        stop=stop_condition,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                ),
                timeout=timeout,
            )
            break
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                await asyncio.sleep(1)
            else:
                logger.error(f"Failed after {max_retries} attempts due to timeout")
                return []
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error on attempt {attempt + 1}: {e}, retrying...")
                await asyncio.sleep(1)
            else:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                return []

    # Parse responses
    completions = []
    for seq in result.sequences:
        message, _ = renderer.parse_response(seq.tokens)
        content = message.get("content", "")
        if isinstance(content, list):
            # Handle structured content (thinking blocks, etc.)
            text_parts = []
            for p in content:
                if p["type"] == "thinking":
                    text_parts.append(f"<think>{p['thinking']}</think>")
                elif p["type"] == "text":
                    text_parts.append(p["text"])
            content = "".join(text_parts)
        completions.append((content, seq.tokens))

    return completions


async def evaluate_problem(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer,
    problem_id: int,
    problem: str,
    reference_answer: str,
    num_samples: int,
    max_tokens: int,
    temperature: float,
) -> ProblemResult | None:
    """Evaluate a single problem with multiple samples."""
    completions = await sample_completions(
        sampling_client,
        renderer,
        problem,
        num_samples,
        max_tokens,
        temperature,
    )

    if not completions:
        logger.warning(f"Problem {problem_id}: No completions returned (timeout/error)")
        return None

    samples = []
    sample_token_counts = []
    for response, tokens in completions:
        predicted = extract_answer_from_response(response)
        correct = grade_response(response, reference_answer)
        total_tokens = len(tokens)
        thinking_tokens = count_thinking_tokens(response, tokenizer)

        samples.append(
            SampleResult(
                response=response,
                predicted_answer=predicted,
                correct=correct,
                total_tokens=total_tokens,
                thinking_tokens=thinking_tokens,
            )
        )
        sample_token_counts.append((total_tokens, thinking_tokens, correct))

    pass_rate = sum(1 for s in samples if s.correct) / len(samples)
    mean_tokens = sum(s.total_tokens for s in samples) / len(samples)
    mean_thinking_tokens = sum(s.thinking_tokens for s in samples) / len(samples)

    # Log running stats
    correct_str = "".join("✓" if c else "✗" for _, _, c in sample_token_counts)
    token_strs = [f"{t}({th})" for t, th, _ in sample_token_counts]
    logger.info(
        f"Problem {problem_id}: {correct_str} | tokens(think): {', '.join(token_strs)} | "
        f"mean={mean_tokens:.0f}, pass={pass_rate:.0%}"
    )

    return ProblemResult(
        problem_id=problem_id,
        problem=problem,
        reference_answer=reference_answer,
        samples=samples,
        pass_rate=pass_rate,
        mean_tokens=mean_tokens,
        mean_thinking_tokens=mean_thinking_tokens,
    )


async def run_evaluation(
    model_name: str,
    checkpoint_path: str | None = None,
    num_problems: int = 100,
    samples_per_problem: int = 4,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    base_url: str | None = None,
    renderer_name: str | None = None,
    concurrency: int = 32,
) -> EvalResults:
    """Run full evaluation on GSM-8K problems."""
    # Setup
    service_client = tinker.ServiceClient(base_url=base_url)

    if checkpoint_path:
        sampling_client = service_client.create_sampling_client(checkpoint_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = get_tokenizer(model_name)
    renderer_name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Load fixed problems
    dataset = get_fixed_gsm8k_problems(num_problems)
    logger.info(f"Evaluating on {len(dataset)} problems with {samples_per_problem} samples each (concurrency={concurrency})")

    # Build tasks for all problems
    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(dataset), desc="Evaluating")

    async def evaluate_with_semaphore(idx: int, row: dict) -> ProblemResult | None:
        async with semaphore:
            problem = row["question"]
            try:
                reference_answer = extract_gsm8k_final_answer(row["answer"])
            except ValueError:
                logger.warning(f"Could not extract reference answer for problem {idx}")
                pbar.update(1)
                return None

            result = await evaluate_problem(
                sampling_client,
                renderer,
                tokenizer,
                idx,
                problem,
                reference_answer,
                samples_per_problem,
                max_tokens,
                temperature,
            )
            pbar.update(1)
            return result

    # Submit all problems concurrently
    tasks = [evaluate_with_semaphore(idx, row) for idx, row in enumerate(dataset)]
    results = await asyncio.gather(*tasks)
    pbar.close()

    # Filter out None results
    problems = [r for r in results if r is not None]

    # Compute aggregate metrics
    all_tokens = [s.total_tokens for p in problems for s in p.samples]
    all_tokens_sorted = sorted(all_tokens)

    accuracy = sum(p.pass_rate for p in problems) / len(problems) if problems else 0.0
    mean_tokens = sum(all_tokens) / len(all_tokens) if all_tokens else 0.0
    median_tokens = all_tokens_sorted[len(all_tokens) // 2] if all_tokens else 0.0
    p90_idx = int(len(all_tokens) * 0.9)
    p90_tokens = all_tokens_sorted[p90_idx] if all_tokens else 0.0
    mean_thinking_tokens = (
        sum(p.mean_thinking_tokens for p in problems) / len(problems) if problems else 0.0
    )
    efficiency = accuracy / (mean_tokens / 100) if mean_tokens > 0 else 0.0

    return EvalResults(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        num_problems=len(problems),
        samples_per_problem=samples_per_problem,
        problems=problems,
        accuracy=accuracy,
        mean_tokens=mean_tokens,
        median_tokens=median_tokens,
        p90_tokens=p90_tokens,
        mean_thinking_tokens=mean_thinking_tokens,
        efficiency=efficiency,
    )


def print_results_table(results: EvalResults):
    """Print a summary table of the evaluation results."""
    print("\n" + "=" * 60)
    print(f"Model: {results.model_name}")
    if results.checkpoint_path:
        print(f"Checkpoint: {results.checkpoint_path}")
    print(f"Problems: {results.num_problems}, Samples per problem: {results.samples_per_problem}")
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {results.accuracy:>14.1%}")
    print(f"{'Mean Tokens':<25} {results.mean_tokens:>15.1f}")
    print(f"{'Median Tokens':<25} {results.median_tokens:>15.1f}")
    print(f"{'P90 Tokens':<25} {results.p90_tokens:>15.1f}")
    print(f"{'Mean Thinking Tokens':<25} {results.mean_thinking_tokens:>15.1f}")
    print(f"{'Efficiency Score':<25} {results.efficiency:>15.2f}")
    print("=" * 60)


def compare_results(results_list: list[EvalResults]):
    """Print a comparison table for multiple evaluation results."""
    print("\n" + "=" * 80)
    print("Comparison Table")
    print("=" * 80)

    # Header
    header = f"{'Model':<20} {'Accuracy':>10} {'Mean Tokens':>12} {'Think Tokens':>12} {'Efficiency':>12}"
    print(header)
    print("-" * 80)

    for results in results_list:
        name = results.model_name[:20]
        if results.checkpoint_path:
            # Use checkpoint name if available
            checkpoint_name = results.checkpoint_path.split("/")[-1][:15]
            name = checkpoint_name

        print(
            f"{name:<20} {results.accuracy:>9.1%} {results.mean_tokens:>12.1f} "
            f"{results.mean_thinking_tokens:>12.1f} {results.efficiency:>12.2f}"
        )

    print("=" * 80)


@chz.chz
class CLIConfig:
    """Configuration for evaluation."""

    model_name: str = "Qwen/Qwen3-8B"
    checkpoint_path: str | None = None
    num_problems: int = 10
    samples_per_problem: int = 4
    max_tokens: int = 2048
    temperature: float = 1.0
    base_url: str | None = None
    renderer_name: str | None = None
    output_path: str | None = None

    # Wandb logging
    wandb_project: str | None = "math-efficiency-interview"
    wandb_name: str | None = None  # e.g., "baseline-eval", "sft-eval", "rl-eval"

    # Concurrency
    concurrency: int = 32


async def cli_main(config: CLIConfig):
    """Run evaluation from CLI."""
    logging.basicConfig(level=logging.INFO)

    # Initialize wandb if configured
    if config.wandb_project:
        wandb_name = config.wandb_name or (
            "baseline-eval" if not config.checkpoint_path else "checkpoint-eval"
        )
        wandb.init(
            project=config.wandb_project,
            name=wandb_name,
            config={
                "model_name": config.model_name,
                "checkpoint_path": config.checkpoint_path,
                "num_problems": config.num_problems,
                "samples_per_problem": config.samples_per_problem,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            },
        )

    results = await run_evaluation(
        model_name=config.model_name,
        checkpoint_path=config.checkpoint_path,
        num_problems=config.num_problems,
        samples_per_problem=config.samples_per_problem,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        base_url=config.base_url,
        renderer_name=config.renderer_name,
        concurrency=config.concurrency,
    )

    print_results_table(results)

    # Log to wandb
    if config.wandb_project:
        wandb.log({
            "accuracy": results.accuracy,
            "mean_tokens": results.mean_tokens,
            "median_tokens": results.median_tokens,
            "p90_tokens": results.p90_tokens,
            "mean_thinking_tokens": results.mean_thinking_tokens,
            "efficiency": results.efficiency,
        })
        wandb.finish()

    if config.output_path:
        with open(config.output_path, "w") as f:
            f.write(results.model_dump_json(indent=2))
        print(f"\nResults saved to: {config.output_path}")


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(config))
