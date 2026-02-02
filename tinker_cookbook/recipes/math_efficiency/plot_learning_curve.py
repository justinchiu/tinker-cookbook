"""Plot learning curves from learning_curve.jsonl files using Altair.

Example:
  python -m tinker_cookbook.recipes.math_efficiency.plot_learning_curve \\
    --run iid-8=/tmp/.../rl/learning_curve.jsonl \\
    --run iid-16=/tmp/.../rl_x2/learning_curve.jsonl \\
    --run answer-hint=/tmp/.../rl_answer_hint/learning_curve.jsonl \\
    --output /tmp/math_efficiency_learning_curve.html
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import altair as alt


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path


def _parse_run_specs(run_args: Iterable[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for raw in run_args:
        if "=" not in raw:
            raise ValueError(f"--run must be LABEL=PATH, got: {raw}")
        label, path_str = raw.split("=", 1)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Learning curve not found: {path}")
        specs.append(RunSpec(label=label, path=path))
    if not specs:
        raise ValueError("At least one --run is required.")
    return specs


def _infer_step_for_final(steps: list[int]) -> int:
    if not steps:
        return 0
    steps_sorted = sorted(steps)
    if len(steps_sorted) == 1:
        return steps_sorted[0] + 1
    deltas = [b - a for a, b in zip(steps_sorted, steps_sorted[1:]) if b > a]
    step_delta = min(deltas) if deltas else 1
    return steps_sorted[-1] + step_delta


def _load_rows(spec: RunSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_rows = [json.loads(line) for line in spec.path.read_text().splitlines() if line]
    numeric_steps = [
        int(row["checkpoint_step"])
        for row in raw_rows
        if row.get("checkpoint_step") is not None
    ]
    final_step = _infer_step_for_final(numeric_steps)
    for row in raw_rows:
        step_val = row.get("checkpoint_step")
        step = int(step_val) if step_val is not None else final_step
        rows.append(
            {
                "run": spec.label,
                "checkpoint": row.get("checkpoint_name", ""),
                "step": step,
                "accuracy_pct": float(row["accuracy"]) * 100.0,
                "mean_tokens": float(row["mean_tokens"]),
                "mean_thinking_tokens": float(row["mean_thinking_tokens"]),
                "efficiency": float(row["efficiency"]),
            }
        )
    return rows


def _build_chart(rows: list[dict[str, object]], title: str) -> alt.Chart:
    alt.data_transformers.disable_max_rows()
    base = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("step:Q", title="Checkpoint step"),
            color=alt.Color("run:N", title="Run"),
            tooltip=[
                "run:N",
                "checkpoint:N",
                alt.Tooltip("step:Q", format=".0f"),
                alt.Tooltip("accuracy_pct:Q", title="Accuracy (%)", format=".3f"),
                alt.Tooltip("mean_tokens:Q", format=".1f"),
                alt.Tooltip("mean_thinking_tokens:Q", format=".1f"),
                alt.Tooltip("efficiency:Q", format=".3f"),
            ],
        )
    )

    acc = base.encode(y=alt.Y("accuracy_pct:Q", title="Accuracy (%)")).properties(
        title="Accuracy vs Step"
    )
    tokens = base.encode(y=alt.Y("mean_tokens:Q", title="Mean Tokens")).properties(
        title="Mean Tokens vs Step"
    )
    return alt.vconcat(acc, tokens).properties(title=title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curves with Altair.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as LABEL=PATH to learning_curve.jsonl (repeatable).",
    )
    parser.add_argument(
        "--output",
        default="learning_curve.html",
        help="Output HTML path for the chart.",
    )
    parser.add_argument("--title", default="Learning Curves", help="Chart title.")
    args = parser.parse_args()

    specs = _parse_run_specs(args.run)
    rows: list[dict[str, object]] = []
    for spec in specs:
        rows.extend(_load_rows(spec))

    chart = _build_chart(rows, title=args.title)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output_path.as_posix())
    print(f"Wrote chart to {output_path}")


if __name__ == "__main__":
    main()
