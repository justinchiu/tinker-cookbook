"""
RL training script for tau2 environments.

Run with:
python -m tinker_cookbook.recipes.taubench.train
"""

# MUST configure tau2 logging BEFORE any tau2 imports
from tinker_cookbook.recipes.taubench import tau2_logging_config

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import cast

import chz
import tinker
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.recipes.taubench.components import (
    AskSonnetMode,
    EpsilonAskSonnetPolicy,
    ExplorationMode,
    RolloutLogger,
)
from tinker_cookbook.recipes.taubench.env import Tau2DatasetBuilder, Tau2Domain
from tinker_cookbook.rl import train

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer_name: str | None = None
    lora_rank: int = 32
    group_size: int = 8
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-5
    max_tokens: int = 4096  # Tau2 conversations can be longer than 20 questions
    eval_every: int = 10  # Less frequent evals
    save_every: int = 10
    wandb_project: str | None = "tau2-rl"
    wandb_name: str | None = None
    log_path: str | None = None
    domain: str = "retail"
    test_group_size: int = 1  # Much smaller samples for evaluations
    load_checkpoint_path: str | None = None
    eval_temperature: float = 0.0  # Greedy decoding for evaluation
    # External LLM (ask_sonnet) configuration
    external_llm_model: str | None = None  # e.g., "claude-sonnet-4-5-20250929"
    external_llm_temperature: float = 0.0
    external_llm_max_tokens: int = 1024
    ask_sonnet_penalty: float = 0.0  # Penalty per ask_sonnet call
    ask_sonnet_mode: AskSonnetMode = AskSonnetMode.CONDITIONING
    # User simulator LLM (default: gpt-4.1, can use gpt-4o-mini for cheaper/faster)
    user_llm: str = "gpt-4.1"
    # Optional token/cost penalties (applied to final reward)
    sonnet_token_penalty_per_1k: float = 0.0
    tau2_user_token_penalty_per_1k: float = 0.0
    tau2_user_cost_penalty: float = 0.0
    # Effective cost reward (replaces old penalty system when set)
    effective_cost_budget: float | None = None
    sonnet_cost_multiplier: float = 30.0
    # Advantage normalization
    normalize_advantages: bool = False
    # Epsilon-greedy policy for ask_sonnet exploration
    epsilon_ask_sonnet: bool = False
    initial_epsilon: float = 0.3
    final_epsilon: float = 0.05
    epsilon_decay_steps: int = 100
    epsilon_decay_type: str = "linear"
    epsilon_seed: int = 42
    exploration_mode: ExplorationMode = ExplorationMode.EPSILON_GREEDY


def build_config(
    cli_config: CLIConfig,
) -> tuple[train.Config, EpsilonAskSonnetPolicy | None]:
    """Build training config and optionally create epsilon policy."""
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

    # Create rollout loggers for conversation logging
    train_rollout_logger = RolloutLogger(
        log_dir=log_path,
        enabled=True,
        subdir="rollouts",
        max_success_per_iter=0,
        max_failure_per_iter=0,
    )
    eval_rollout_logger = RolloutLogger(
        log_dir=log_path,
        enabled=True,
        subdir="eval_rollouts",
        max_success_per_iter=0,
        max_failure_per_iter=0,
    )

    dataset_builder = Tau2DatasetBuilder(
        batch_size=cli_config.batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        domain=cast(Tau2Domain, cli_config.domain),
        test_group_size=cli_config.test_group_size,
        num_epochs=cli_config.num_epochs,
        external_llm_model=cli_config.external_llm_model,
        external_llm_temperature=cli_config.external_llm_temperature,
        external_llm_max_tokens=cli_config.external_llm_max_tokens,
        ask_sonnet_mode=cli_config.ask_sonnet_mode,
        user_llm=cli_config.user_llm,
        ask_sonnet_penalty=cli_config.ask_sonnet_penalty,
        sonnet_token_penalty_per_1k=cli_config.sonnet_token_penalty_per_1k,
        tau2_user_token_penalty_per_1k=cli_config.tau2_user_token_penalty_per_1k,
        tau2_user_cost_penalty=cli_config.tau2_user_cost_penalty,
        effective_cost_budget=cli_config.effective_cost_budget,
        sonnet_cost_multiplier=cli_config.sonnet_cost_multiplier,
        rollout_logger=train_rollout_logger,
        eval_rollout_logger=eval_rollout_logger,
    )

    # Create epsilon policy if enabled
    from collections.abc import Callable

    epsilon_policy: EpsilonAskSonnetPolicy | None = None
    _policy_factory: train.PolicyFactory | None = None
    _on_train_step: Callable[[int], None] | None = None

    if cli_config.epsilon_ask_sonnet:
        epsilon_policy = EpsilonAskSonnetPolicy(
            model_name=model_name,
            max_tokens=cli_config.max_tokens,
            temperature=1.0,
            initial_epsilon=cli_config.initial_epsilon,
            final_epsilon=cli_config.final_epsilon,
            decay_steps=cli_config.epsilon_decay_steps,
            decay_type=cli_config.epsilon_decay_type,
            seed=cli_config.epsilon_seed,
            mode=cli_config.exploration_mode,
        )

        _ep = epsilon_policy  # capture for closures

        def _make_policy_factory(
            sampling_client: tinker.SamplingClient,
        ) -> TokenCompleter:
            _ep.sampling_client = sampling_client
            return _ep

        def _make_on_train_step(step: int) -> None:
            _ep.step()
            if step % 10 == 0:
                metrics = _ep.get_metrics_and_reset()
                logger.info(
                    f"Epsilon policy step {step}: epsilon={metrics['epsilon_policy/current_epsilon']:.3f}, "
                    f"forced={metrics['epsilon_policy/forced_ask_sonnet_total']}, "
                    f"policy_ask={metrics['epsilon_policy/policy_ask_sonnet_total']}, "
                    f"policy_other={metrics['epsilon_policy/policy_other_total']}"
                )

        _policy_factory = _make_policy_factory
        _on_train_step = _make_on_train_step

        logger.info(
            f"Epsilon ask_sonnet enabled: mode={cli_config.exploration_mode.value}, "
            f"initial_epsilon={cli_config.initial_epsilon}, final_epsilon={cli_config.final_epsilon}, "
            f"epsilon_decay_steps={cli_config.epsilon_decay_steps}"
        )

    config = train.Config(
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
        eval_temperature=cli_config.eval_temperature,
        normalize_advantages=cli_config.normalize_advantages,
        policy_factory=_policy_factory,
        on_train_step=_on_train_step,
    )
    return config, epsilon_policy


async def async_main(config: train.Config):
    """Async wrapper that sets thread pool size before training."""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=64))
    await train.main(config)


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config, _epsilon_policy = build_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    tau2_logging_config.setup_tau2_logging()
    asyncio.run(async_main(config))


if __name__ == "__main__":
    main()
