"""
Supervised fine-tuning on tau2 simulation data.

Usage:
    python -m tinker_cookbook.recipes.taubench.train_sft

Or with custom parameters:
    python -m tinker_cookbook.recipes.taubench.train_sft \
        model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
        learning_rate=5e-5 \
        num_epochs=5
"""

import asyncio
from datetime import datetime
from typing import cast

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.taubench.components import AskSonnetMode
from tinker_cookbook.recipes.taubench.env import build_tau_eval_builders
from tinker_cookbook.recipes.taubench.sft_dataset import (
    Tau2SimulationFilesBuilder,
)
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule


@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer_name: str | None = None
    lora_rank: int = 32

    # Training parameters
    learning_rate: float = 3e-5
    lr_schedule: str = "cosine_warmup"
    num_epochs: int = 3
    batch_size: int = 64
    max_length: int | None = 25000  # Tau2 conversations can be long

    # Training behavior
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES

    # Checkpointing and evaluation
    save_every: int = 20
    eval_every: int = 20
    infrequent_eval_every: int = 41
    load_checkpoint_path: str | None = None

    # Tau2 rollout evaluation parameters
    enable_tau_eval: bool = True
    eval_domain: str = "retail"
    eval_num_tasks: int | None = None
    eval_group_size: int = 1
    eval_batch_size: int = 3
    eval_max_tokens: int = 1024
    eval_temperature: float = 0.0
    eval_task_seed: int = 0
    eval_name: str = "tau2_rollout"
    eval_max_context_length: int | None = 16384

    # Infrastructure
    base_url: str | None = None

    # Logging
    log_path: str | None = None
    wandb_project: str = "tau2-sft"
    wandb_name: str = "qwen3-30b-a3b-instruct-alldomains"
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # ask_sonnet injection
    ask_sonnet_injection_rate: float = 0.0
    train_on_ask_sonnet_only: bool = False

    # External LLM (ask_sonnet) for eval
    external_llm_model: str | None = None
    external_llm_temperature: float = 0.0
    external_llm_max_tokens: int = 1024
    ask_sonnet_mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION


def main():
    cli_config = chz.entrypoint(CLIConfig)

    # Default simulation files
    SIMULATION_FILES = [
        "/home/ubuntu/code/tau2-bench/data/simulations/2025-11-22T13:36:02.147814_telecom_llm_agent_claude-sonnet-4-5-20250929_user_simulator_gpt-4.1.json",
        "/home/ubuntu/code/tau2-bench/data/simulations/2025-11-22T22:11:21.085044_airline_llm_agent_claude-sonnet-4-5-20250929_user_simulator_gpt-4.1.json",
        "/home/ubuntu/code/tau2-bench/data/simulations/2025-11-22T22:11:36.713728_retail_llm_agent_claude-sonnet-4-5-20250929_user_simulator_gpt-4.1.json",
        "/home/ubuntu/code/tinker-cookbook/data/tau2_rollouts_sonnet/airline_16trials_sonnet45.json",
        "/home/ubuntu/code/tinker-cookbook/data/tau2_rollouts_sonnet/retail_16trials_sonnet45.json",
        "/home/ubuntu/code/tinker-cookbook/data/tau2_rollouts_sonnet/telecom_16trials_sonnet45.json",
    ]

    model_name_safe = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"tau2-sft-multi-domain-{model_name_safe}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{date_and_time}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"logs/tinker-examples/tau2-sft/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    cli_utils.check_log_dir(
        log_path,
        behavior_if_exists=cli_config.behavior_if_log_dir_exists,
    )

    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    train_on_what = cli_config.train_on_what
    if cli_config.train_on_ask_sonnet_only:
        train_on_what = TrainOnWhat.CUSTOMIZED
        print(
            "train_on_ask_sonnet_only=True: using CUSTOMIZED train_on_what, "
            "only ask_sonnet turns will have loss"
        )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
        train_on_what=train_on_what,
    )

    dataset_builder = Tau2SimulationFilesBuilder(
        common_config=common_config,
        simulation_files=SIMULATION_FILES,
        ask_sonnet_injection_rate=cli_config.ask_sonnet_injection_rate,
        ask_sonnet_injection_mode=cli_config.ask_sonnet_mode,
        train_on_ask_sonnet_only=cli_config.train_on_ask_sonnet_only,
    )

    external_llm_model = cli_config.external_llm_model
    if cli_config.ask_sonnet_injection_rate > 0 and external_llm_model is None:
        external_llm_model = "claude-sonnet-4-5-20250929"
        print(
            f"ask_sonnet injection enabled, using default external_llm_model: {external_llm_model}"
        )

    tau_eval_builders = build_tau_eval_builders(
        enabled=cli_config.enable_tau_eval,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        domain=cli_config.eval_domain,
        num_tasks=cli_config.eval_num_tasks,
        batch_size=cli_config.eval_batch_size,
        group_size=cli_config.eval_group_size,
        max_tokens=cli_config.eval_max_tokens,
        temperature=cli_config.eval_temperature,
        task_seed=cli_config.eval_task_seed,
        eval_name=cli_config.eval_name,
        max_context_length=cli_config.eval_max_context_length,
        external_llm_model=external_llm_model,
        external_llm_temperature=cli_config.external_llm_temperature,
        external_llm_max_tokens=cli_config.external_llm_max_tokens,
        ask_sonnet_mode=cli_config.ask_sonnet_mode,
        log_dir=log_path,
    )

    config = train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        evaluator_builders=tau_eval_builders,
        infrequent_evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule=cast(LRSchedule, cli_config.lr_schedule),
        num_epochs=cli_config.num_epochs,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        infrequent_eval_every=cli_config.infrequent_eval_every,
    )

    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
