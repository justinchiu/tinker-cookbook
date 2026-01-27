"""
Generate efficient training data via rejection sampling.

For each problem:
1. Sample N completions
2. Grade for correctness
3. Filter to correct answers only
4. Select the shortest correct answer
5. Save as JSONL for SFT training
"""

import asyncio
import json
import logging
from typing import cast

import chz
import tinker
from datasets import Dataset, load_dataset
from tqdm import tqdm

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


def get_fixed_gsm8k_problems(num_problems: int = 100, seed: int = 42) -> Dataset:
    """Load a fixed set of GSM-8K problems for consistent evaluation."""
    ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split="train"))
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(num_problems, len(ds))))


def extract_answer_from_response(response: str) -> str | None:
    """Extract the answer from a model response."""
    try:
        return extract_boxed(response)
    except ValueError:
        return None


def grade_response(response: str, reference_answer: str) -> bool:
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
) -> list[tuple[str, int]]:
    """Sample multiple completions for a problem.

    Returns list of (response_text, token_count) tuples.
    """
    # Build the prompt
    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    messages = [{"role": "user", "content": problem + question_suffix}]
    model_input = renderer.build_generation_prompt(messages)
    stop_condition = renderer.get_stop_sequences()

    # Sample
    result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(
            stop=stop_condition,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    )

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
        completions.append((content, len(seq.tokens)))

    return completions


async def generate_efficient_example(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    problem: str,
    reference_answer: str,
    num_samples: int,
    max_tokens: int,
    temperature: float,
) -> dict | None:
    """Generate an efficient training example for a single problem.

    Returns a dict with 'messages' field containing user and assistant messages,
    or None if no correct answer was found.
    """
    completions = await sample_completions(
        sampling_client,
        renderer,
        problem,
        num_samples,
        max_tokens,
        temperature,
    )

    # Filter to correct responses and find shortest
    correct_completions = []
    for response, token_count in completions:
        if grade_response(response, reference_answer):
            correct_completions.append((response, token_count))

    if not correct_completions:
        return None

    # Sort by token count and select shortest
    correct_completions.sort(key=lambda x: x[1])
    shortest_response, shortest_tokens = correct_completions[0]

    # Build the training example
    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    user_message = {"role": "user", "content": problem + question_suffix}
    assistant_message = {"role": "assistant", "content": shortest_response}

    return {
        "messages": [user_message, assistant_message],
        "metadata": {
            "problem": problem,
            "reference_answer": reference_answer,
            "response_tokens": shortest_tokens,
            "num_correct": len(correct_completions),
            "num_samples": num_samples,
        },
    }


async def generate_efficient_data(
    model_name: str,
    num_problems: int = 100,
    num_samples: int = 16,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    output_path: str = "/tmp/gsm8k_efficient.jsonl",
    base_url: str | None = None,
    renderer_name: str | None = None,
    checkpoint_path: str | None = None,
    concurrency: int = 32,
):
    """Generate efficient training data from GSM-8K problems."""
    # Setup
    service_client = tinker.ServiceClient(base_url=base_url)

    if checkpoint_path:
        sampling_client = service_client.create_sampling_client(checkpoint_path)
        logger.info(f"Using checkpoint: {checkpoint_path}")
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)
        logger.info(f"Using base model: {model_name}")

    tokenizer = get_tokenizer(model_name)
    renderer_name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Load fixed problems
    dataset = get_fixed_gsm8k_problems(num_problems)
    logger.info(
        f"Generating efficient data from {len(dataset)} problems with {num_samples} samples each (concurrency={concurrency})"
    )

    # Build tasks for all problems with semaphore
    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(dataset), desc="Generating efficient data")

    async def generate_with_semaphore(idx: int, row: dict) -> dict | None:
        async with semaphore:
            problem = row["question"]
            try:
                reference_answer = extract_gsm8k_final_answer(row["answer"])
            except ValueError:
                logger.warning(f"Could not extract reference answer for problem {idx}")
                pbar.update(1)
                return None

            example = await generate_efficient_example(
                sampling_client,
                renderer,
                problem,
                reference_answer,
                num_samples,
                max_tokens,
                temperature,
            )

            if example:
                logger.info(
                    f"Problem {idx}: {example['metadata']['num_correct']}/{num_samples} correct, "
                    f"shortest={example['metadata']['response_tokens']} tokens"
                )
            else:
                logger.info(f"Problem {idx}: 0/{num_samples} correct")

            pbar.update(1)
            return example

    # Submit all problems concurrently
    tasks = [generate_with_semaphore(idx, row) for idx, row in enumerate(dataset)]
    results = await asyncio.gather(*tasks)
    pbar.close()

    # Filter out None results
    examples = [r for r in results if r is not None]
    skipped = len(dataset) - len(examples)
    total_tokens_after = sum(e["metadata"]["response_tokens"] for e in examples)

    # Write output
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("Data Generation Summary")
    print("=" * 60)
    print(f"Total problems:        {len(dataset)}")
    print(f"Successful examples:   {len(examples)}")
    print(f"Skipped (no correct):  {skipped}")
    print(f"Success rate:          {len(examples) / len(dataset):.1%}")
    if examples:
        print(f"Mean tokens/example:   {total_tokens_after / len(examples):.1f}")
    print(f"Output file:           {output_path}")
    print("=" * 60)

    return examples


@chz.chz
class CLIConfig:
    """Configuration for data generation."""

    model_name: str = "Qwen/Qwen3-8B"
    checkpoint_path: str | None = None
    num_problems: int = 10
    num_samples: int = 16
    max_tokens: int = 4096
    temperature: float = 1.0
    output_path: str = "/tmp/gsm8k_efficient.jsonl"
    base_url: str | None = None
    renderer_name: str | None = None
    concurrency: int = 32


async def cli_main(config: CLIConfig):
    """Run data generation from CLI."""
    logging.basicConfig(level=logging.INFO)

    await generate_efficient_data(
        model_name=config.model_name,
        num_problems=config.num_problems,
        num_samples=config.num_samples,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        output_path=config.output_path,
        base_url=config.base_url,
        renderer_name=config.renderer_name,
        checkpoint_path=config.checkpoint_path,
        concurrency=config.concurrency,
    )


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(config))
