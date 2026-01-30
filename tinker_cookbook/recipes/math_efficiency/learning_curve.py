"""
Evaluate checkpoints over training steps to build learning curves.
"""

import asyncio
import json
import logging
import os
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import chz

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.recipes.math_efficiency.eval import run_evaluation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricSeries:
    steps: list[int]
    cumulative_tokens: list[int]
    cumulative_time: list[float]

    def cumulative_at(self, step: int) -> tuple[int | None, float | None]:
        if not self.steps:
            return None, None
        idx = bisect_right(self.steps, step) - 1
        if idx < 0:
            return None, None
        return self.cumulative_tokens[idx], self.cumulative_time[idx]


def _load_metric_series(log_path: str) -> MetricSeries:
    metrics_path = os.path.join(log_path, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        logger.warning("No metrics.jsonl found at %s", metrics_path)
        return MetricSeries(steps=[], cumulative_tokens=[], cumulative_time=[])

    entries: list[dict[str, Any]] = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed metrics line")

    entries = [e for e in entries if isinstance(e.get("step"), int)]
    entries.sort(key=lambda e: e["step"])

    steps: list[int] = []
    cumulative_tokens: list[int] = []
    cumulative_time: list[float] = []
    total_tokens = 0
    total_time = 0.0
    for entry in entries:
        step = entry["step"]
        total_tokens += int(entry.get("num_tokens", 0))
        total_time += float(entry.get("time/total", 0.0))
        steps.append(step)
        cumulative_tokens.append(total_tokens)
        cumulative_time.append(total_time)

    return MetricSeries(steps=steps, cumulative_tokens=cumulative_tokens, cumulative_time=cumulative_time)


def _checkpoint_step(checkpoint: dict[str, Any]) -> int | None:
    if isinstance(checkpoint.get("step"), int):
        return checkpoint["step"]
    if isinstance(checkpoint.get("batch"), int) and isinstance(checkpoint.get("epoch"), int):
        # Fall back to batch index if epoch is present but total steps unknown.
        return checkpoint["batch"]
    name = checkpoint.get("name")
    if isinstance(name, str) and name.isdigit():
        return int(name)
    return None


def _select_checkpoints(
    checkpoints: list[dict[str, Any]],
    stride: int,
    max_checkpoints: int | None,
) -> list[dict[str, Any]]:
    sampler_checkpoints = [c for c in checkpoints if "sampler_path" in c]
    if stride > 1:
        sampler_checkpoints = sampler_checkpoints[::stride]
    if max_checkpoints is not None and len(sampler_checkpoints) > max_checkpoints:
        # Evenly spaced selection across the run
        if max_checkpoints <= 1:
            sampler_checkpoints = [sampler_checkpoints[-1]]
        else:
            span = len(sampler_checkpoints) - 1
            indices = [int(round(i * span / (max_checkpoints - 1))) for i in range(max_checkpoints)]
            sampler_checkpoints = [sampler_checkpoints[i] for i in indices]
    return sampler_checkpoints


def _load_existing(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    seen = set()
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            sampler_path = entry.get("sampler_path")
            if isinstance(sampler_path, str) and sampler_path:
                seen.add(sampler_path)
                continue
            name = entry.get("checkpoint_name")
            if isinstance(name, str) and name:
                seen.add(name)
    return seen


@chz.chz
class CLIConfig:
    log_path: str
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = None
    eval_num_problems: int = 10
    eval_samples_per_problem: int = 4
    max_tokens: int = 4096
    temperature: float = 1.0
    base_url: str | None = None
    checkpoint_stride: int = 1
    max_checkpoints: int | None = None
    output_path: str | None = None
    resume: bool = True


async def cli_main(config: CLIConfig) -> None:
    checkpoints = checkpoint_utils.load_checkpoints_file(config.log_path)
    if not checkpoints:
        logger.error("No checkpoints found under %s", config.log_path)
        return

    checkpoints = _select_checkpoints(
        checkpoints, stride=config.checkpoint_stride, max_checkpoints=config.max_checkpoints
    )
    if not checkpoints:
        logger.error("No sampler checkpoints available after filtering")
        return

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    output_path = config.output_path or os.path.join(config.log_path, "learning_curve.jsonl")
    seen = _load_existing(output_path) if config.resume else set()

    metric_series = _load_metric_series(config.log_path)

    for checkpoint in checkpoints:
        name = str(checkpoint.get("name", ""))
        sampler_path = checkpoint["sampler_path"]
        key = sampler_path or name
        if key in seen:
            logger.info("Skipping checkpoint %s (already evaluated)", name)
            continue
        step = _checkpoint_step(checkpoint)
        train_tokens_cum, train_time_cum = (None, None)
        if step is not None:
            train_tokens_cum, train_time_cum = metric_series.cumulative_at(step)

        logger.info("Evaluating checkpoint %s", name)
        results = await run_evaluation(
            model_name=config.model_name,
            checkpoint_path=sampler_path,
            num_problems=config.eval_num_problems,
            samples_per_problem=config.eval_samples_per_problem,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            base_url=config.base_url,
            renderer_name=renderer_name,
        )

        entry = {
            "checkpoint_name": name,
            "checkpoint_step": step,
            "sampler_path": sampler_path,
            "accuracy": results.accuracy,
            "mean_tokens": results.mean_tokens,
            "mean_thinking_tokens": results.mean_thinking_tokens,
            "efficiency": results.efficiency,
            "train_tokens_cum": train_tokens_cum,
            "train_time_cum": train_time_cum,
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Wrote learning curve entry to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
