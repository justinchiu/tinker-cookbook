"""
RL training script for efficient math reasoning.

Uses EfficientMathEnv which rewards correct AND short answers.
Reward: correct * -num_tokens (GRPO-style advantage normalization handles the rest)
Runs evaluation automatically after training completes.

Based on compute-optimal RL scaling recommendations:
- Scale group_size (n) as primary lever
- LR scales with sqrt(batch_size): lr = lr_base * sqrt(B / 1024)
- No KL penalty for hard problems
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import chz
from tinker.types import LossFnType

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_efficiency.efficient_env import (
    EfficientGsm8kDatasetBuilder,
)
from tinker_cookbook.recipes.math_efficiency.eval import (
    print_results_table,
    run_evaluation,
)
from tinker_cookbook.rl.train import Config, StreamMinibatchConfig, main

logger = logging.getLogger(__name__)


def compute_scaled_lr(base_lr: float, batch_size: int, reference_batch: int = 1024) -> float:
    """Compute scaled learning rate based on batch size.

    Following compute-optimal RL scaling: lr = lr_base * sqrt(B / reference_batch)
    """
    import math

    return base_lr * math.sqrt(batch_size / reference_batch)


@chz.chz
class CLIConfig:
    """Configuration for efficient RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    num_problems: int = 10  # Fixed set of GSM-8K problems
    seed: int = 42  # Seed for problem selection
    n_epochs: int = 1  # Number of epochs (passes through the dataset)

    # RL hyperparameters (based on compute-optimal scaling)
    group_size: int = 8  # n: parallel rollouts per problem (primary lever)
    groups_per_batch: int = 32  # B_problem: problems per batch

    # Learning rate - scaled automatically based on batch size
    # Base LR at batch=1024: 1e-6, we compute scaled version
    learning_rate: float | None = None  # If None, computed from base_lr
    base_lr: float = 1e-6  # Base LR at batch_size=1024

    # Sampling parameters
    max_tokens: int = 4096
    temperature: float = 1.0

    # No KL penalty for hard problems (per compute-optimal scaling)
    kl_penalty_coef: float = 0.0

    # Number of optimizer steps per training iteration
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = "math-efficiency-interview"
    wandb_name: str | None = None  # Defaults to "rl-efficient-{model}"
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 10

    # Checkpointing
    save_every: int = 10

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Loss function
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Minibatch streaming for overlapping sampling and training
    num_minibatches: int | None = None  # If set, enable streaming

    # Advantage normalization (standardize to mean=0, std=1)
    normalize_advantages: bool = True  # Recommended for length-based rewards

    # Final evaluation config
    run_final_eval: bool = True
    eval_num_problems: int = 10
    eval_samples_per_problem: int = 4


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""
    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Compute total batch size and scaled learning rate
    total_batch_size = cli_config.group_size * cli_config.groups_per_batch

    if cli_config.learning_rate is not None:
        learning_rate = cli_config.learning_rate
    else:
        learning_rate = compute_scaled_lr(cli_config.base_lr, total_batch_size)

    logger.info(f"Batch size: {total_batch_size} (group_size={cli_config.group_size} x groups_per_batch={cli_config.groups_per_batch})")
    logger.info(f"Scaled learning rate: {learning_rate:.2e}")

    # Generate log path and run name
    model_name_safe = cli_config.model_name.replace("/", "-")
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    run_name = f"rl-{model_name_safe}-{timestamp}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/math_efficiency_rl/{run_name}"

    wandb_name = cli_config.wandb_name or run_name

    # Create dataset builder
    dataset_builder = EfficientGsm8kDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        num_problems=cli_config.num_problems,
        seed=cli_config.seed,
        n_epochs=cli_config.n_epochs,
    )

    # Create streaming config if requested
    stream_minibatch_config = None
    if cli_config.num_minibatches is not None:
        stream_minibatch_config = StreamMinibatchConfig(
            groups_per_batch=cli_config.groups_per_batch,
            num_minibatches=cli_config.num_minibatches,
        )

    # Create full config
    config = Config(
        learning_rate=learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        stream_minibatch_config=stream_minibatch_config,
        # Remove constant reward groups to focus on informative samples
        remove_constant_reward_groups=False,
        # Normalize advantages for length-based rewards
        normalize_advantages=cli_config.normalize_advantages,
    )

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    logger.info(f"Starting RL training with efficiency reward")
    logger.info(f"Log path: {log_path}")
    logger.info(f"Problems: {cli_config.num_problems}, Group size: {cli_config.group_size}, Epochs: {cli_config.n_epochs}")

    # Run training
    await main(config)

    # Run final evaluation
    if cli_config.run_final_eval:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Running final evaluation on trained model")
        logger.info("=" * 60)

        # Read checkpoint path from checkpoints.jsonl
        checkpoints_file = os.path.join(log_path, "checkpoints.jsonl")
        checkpoint_path = None
        if os.path.exists(checkpoints_file):
            import json
            with open(checkpoints_file, "r") as f:
                for line in f:
                    ckpt = json.loads(line)
                    if ckpt.get("name") == "final":
                        checkpoint_path = ckpt.get("sampler_path")
                        break

        if not checkpoint_path:
            logger.error("Could not find final checkpoint path")
            return

        eval_output_path = os.path.join(log_path, "eval_results.json")

        eval_results = await run_evaluation(
            model_name=cli_config.model_name,
            checkpoint_path=checkpoint_path,
            num_problems=cli_config.eval_num_problems,
            samples_per_problem=cli_config.eval_samples_per_problem,
            max_tokens=cli_config.max_tokens,
            temperature=cli_config.temperature,
            base_url=cli_config.base_url,
            renderer_name=renderer_name,
        )

        print_results_table(eval_results)

        # Save results
        with open(eval_output_path, "w") as f:
            f.write(eval_results.model_dump_json(indent=2))
        logger.info(f"Evaluation results saved to: {eval_output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
