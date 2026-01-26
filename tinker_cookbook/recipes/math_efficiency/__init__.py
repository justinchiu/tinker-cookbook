"""
Math Efficiency Training Recipe

Train models to solve math problems with fewer reasoning tokens while maintaining accuracy.

Two training methods:
1. Rejection-Sampled SFT (Method 1):
   - Sample multiple completions, keep shortest correct ones
   - Fine-tune on efficient examples

2. Online RL with Efficiency Reward (Method 2):
   - Reward correct answers proportionally to how short they are
   - Self-improving: target gets harder as model finds shorter solutions

Scripts:
- eval.py: Evaluate accuracy, token counts, thinking tokens, efficiency
- generate_efficient_data.py: Generate efficient training data via rejection sampling
- train_sft.py: SFT on efficient examples
- efficient_env.py: Environment with efficiency reward
- train_rl.py: RL training with efficiency reward
"""

from tinker_cookbook.recipes.math_efficiency.efficient_env import (
    EfficientGsm8kDataset,
    EfficientGsm8kDatasetBuilder,
    EfficientMathEnv,
    EfficientProblemGroupBuilder,
)
from tinker_cookbook.recipes.math_efficiency.eval import (
    EvalResults,
    ProblemResult,
    SampleResult,
    compare_results,
    print_results_table,
    run_evaluation,
)

__all__ = [
    "EfficientMathEnv",
    "EfficientGsm8kDataset",
    "EfficientGsm8kDatasetBuilder",
    "EfficientProblemGroupBuilder",
    "run_evaluation",
    "EvalResults",
    "ProblemResult",
    "SampleResult",
    "print_results_table",
    "compare_results",
]
