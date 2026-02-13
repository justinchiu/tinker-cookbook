"""
One-off evaluation script for tau2 environments.

Usage:
    python -m tinker_cookbook.recipes.taubench.eval \
        checkpoint_path=tinker://your-checkpoint-path \
        domain=retail
"""

# MUST configure tau2 logging BEFORE any tau2 imports
from tinker_cookbook.recipes.taubench import tau2_logging_config

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import chz
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.recipes.taubench.components import AskSonnetMode, RolloutLogger
from tinker_cookbook.recipes.taubench.env import Tau2Dataset, Tau2DatasetBuilder
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.utils import logtree, ml_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    checkpoint_path: str | None = None
    lora_rank: int = 32

    domain: str = "retail"
    num_tasks: int | None = None
    batch_size: int = 8
    group_size: int = 1
    max_tokens: int = 4096
    temperature: float = 0.0
    task_seed: int = 0
    max_context_length: int | None = 16384

    log_path: str | None = None
    num_groups_to_log: int = 4
    wandb_project: str | None = None
    wandb_name: str | None = None

    renderer_name: str | None = None

    external_llm_model: str | None = None
    external_llm_temperature: float = 0.0
    external_llm_max_tokens: int = 1024
    ask_sonnet_mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION


async def run_evaluation(config: CLIConfig) -> dict[str, float]:
    """Run tau2 evaluation and return metrics."""
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )

    logger.info("=" * 60)
    logger.info("TAU2 EVALUATION")
    logger.info("=" * 60)
    logger.info("Model: %s", config.model_name)
    logger.info("Checkpoint: %s", config.checkpoint_path or "(base model)")
    logger.info("Domain: %s", config.domain)
    logger.info("Renderer: %s", renderer_name)
    logger.info("Temperature: %s", config.temperature)
    logger.info("Max tokens: %s", config.max_tokens)
    if config.external_llm_model:
        logger.info(
            "External LLM: %s (mode=%s)",
            config.external_llm_model,
            config.ask_sonnet_mode.name,
        )
    logger.info("=" * 60)

    dataset_builder = Tau2DatasetBuilder(
        batch_size=config.batch_size,
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        group_size=config.group_size,
        domain=config.domain,
        seed=config.task_seed,
        test_group_size=config.group_size,
        num_epochs=1,
        external_llm_model=config.external_llm_model,
        external_llm_temperature=config.external_llm_temperature,
        external_llm_max_tokens=config.external_llm_max_tokens,
        ask_sonnet_mode=config.ask_sonnet_mode,
    )

    _, raw_test_dataset = await dataset_builder()

    tasks = list(raw_test_dataset.tasks)
    if config.num_tasks is not None:
        tasks = tasks[: config.num_tasks]

    if not tasks:
        raise ValueError(f"No tasks found for domain={config.domain}")

    logger.info("Evaluating on %d tasks", len(tasks))

    rollout_logger = (
        RolloutLogger(log_dir=config.log_path, enabled=bool(config.log_path))
        if config.log_path
        else None
    )

    test_dataset = Tau2Dataset(
        tasks=tasks,
        renderer=raw_test_dataset.renderer,
        domain=raw_test_dataset.domain,
        batch_size=min(len(tasks), config.batch_size),
        group_size=config.group_size,
        max_context_length=config.max_context_length,
        external_llm_model=config.external_llm_model,
        external_llm_temperature=config.external_llm_temperature,
        external_llm_max_tokens=config.external_llm_max_tokens,
        ask_sonnet_mode=config.ask_sonnet_mode,
        rollout_logger=rollout_logger,
    )

    service_client = tinker.ServiceClient()

    if config.checkpoint_path:
        logger.info("Loading checkpoint: %s", config.checkpoint_path)
        training_client = await service_client.create_training_client_from_state_async(
            config.checkpoint_path,
            user_metadata={},
        )
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name="eval"
        )
    else:
        logger.info("Using base model (no checkpoint)")
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
        )
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name="eval"
        )

    evaluator = RLTestSetEvaluator(
        dataset=test_dataset,
        max_tokens=config.max_tokens,
        name="tau2_eval",
        num_groups_to_log=config.num_groups_to_log,
    )

    logger.info("Starting evaluation...")

    if config.log_path:
        logtree_path = os.path.join(config.log_path, "eval_rollouts.html")
        os.makedirs(config.log_path, exist_ok=True)
        with logtree.init_trace("tau2_eval", path=logtree_path):
            metrics = await evaluator(sampling_client)
    else:
        metrics = await evaluator(sampling_client)

    return metrics


def main():
    cli_config = chz.entrypoint(CLIConfig)
    tau2_logging_config.setup_tau2_logging()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_default_executor(ThreadPoolExecutor(max_workers=64))

    try:
        metrics = loop.run_until_complete(run_evaluation(cli_config))
    finally:
        loop.close()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)

    if cli_config.log_path:
        os.makedirs(cli_config.log_path, exist_ok=True)
        metrics_path = os.path.join(cli_config.log_path, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", metrics_path)

        config_dict = {
            "model_name": cli_config.model_name,
            "checkpoint_path": cli_config.checkpoint_path,
            "domain": cli_config.domain,
            "num_tasks": cli_config.num_tasks,
            "batch_size": cli_config.batch_size,
            "group_size": cli_config.group_size,
            "max_tokens": cli_config.max_tokens,
            "temperature": cli_config.temperature,
            "task_seed": cli_config.task_seed,
            "max_context_length": cli_config.max_context_length,
            "external_llm_model": cli_config.external_llm_model,
            "external_llm_temperature": cli_config.external_llm_temperature,
            "external_llm_max_tokens": cli_config.external_llm_max_tokens,
            "ask_sonnet_mode": cli_config.ask_sonnet_mode.name,
            "timestamp": datetime.now().isoformat(),
        }
        config_path = os.path.join(cli_config.log_path, "eval_config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info("Saved config to %s", config_path)

        if cli_config.wandb_project:
            ml_logger = ml_log.setup_logging(
                log_path=cli_config.log_path,
                wandb_project=cli_config.wandb_project,
                wandb_name=cli_config.wandb_name
                or f"tau2-eval-{cli_config.domain}-{datetime.now().strftime('%Y%m%d-%H%M')}",
                config=config_dict,
            )
            ml_logger.log_metrics(metrics, step=0)
            ml_logger.close()
            logger.info("Logged metrics to wandb")

    reward_key = "tau2_eval/env/all/reward/total"
    if reward_key in metrics:
        print(f"\nFinal reward: {metrics[reward_key]:.4f}")

    return metrics


if __name__ == "__main__":
    main()
