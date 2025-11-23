"""
RL training script for tau2 environments.

Run with:
python -m tinker_cookbook.recipes.taubench.train
"""

# MUST configure tau2 logging BEFORE any tau2 imports
from tinker_cookbook.recipes.taubench import tau2_logging_config

import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.taubench.env import Tau2DatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer_name: str | None = None
    lora_rank: int = 32
    group_size: int = 4
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    max_tokens: int = 4096  # Tau2 conversations can be longer than 20 questions
    eval_every: int = 5  # Less frequent evals
    save_every: int = 5
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None
    domain: str = "telecom"
    test_group_size: int = 1  # Much smaller samples for evaluations
    load_checkpoint_path: str | None = None  # Resume from checkpoint (e.g., tinker://...)


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"tau2-{cli_config.domain}-{model_name}-{cli_config.group_size}group-{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/tau2-rl/{run_name}"
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = Tau2DatasetBuilder(
        batch_size=cli_config.batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        domain=cli_config.domain,
        test_group_size=cli_config.test_group_size,
        num_epochs=cli_config.num_epochs,
    )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    # Setup tau2 logging after log directory is validated
    tau2_logging_config.setup_tau2_logging()
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
