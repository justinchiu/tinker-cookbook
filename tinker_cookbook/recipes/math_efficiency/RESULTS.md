# Math Efficiency Training Results

Training Qwen3-8B to solve GSM-8K problems with fewer reasoning tokens while maintaining accuracy.

## Experiment Settings
- **max_tokens**: 4096
- **Timeout handling**: Timeouts count as 0% pass rate (not excluded)
- **Dataset**: 100 fixed problems from GSM-8K train set
- **Samples per problem**: 4 (evaluation)

## Wandb Project
https://wandb.ai/percepta-ai/math-efficiency-interview

## Results Summary

| Model | Accuracy | Mean Tokens | Thinking Tokens | Efficiency | Wandb |
|-------|----------|-------------|-----------------|------------|-------|
| Baseline | 91.0% | 1805 | 1308 | 0.05 | [baseline-eval-v2](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/0zx66j80) |
| SFT | 83.0% | 1278 | 850 | 0.06 | [sft-eval-v2](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/hwkgnpo1) |
| RL (1 epoch) | 59.0% | 1302 | 622 | 0.05 | [rl-train-v2](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/khmkki1q) |

## Baseline Evaluation

**Run:** [baseline-eval-v2](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/0zx66j80)

| Metric | Value |
|--------|-------|
| Accuracy | 91.0% |
| Mean Tokens | 1805 |
| Median Tokens | 1581 |
| P90 Tokens | 3291 |
| Mean Thinking Tokens | 1308 (72% of total) |
| Efficiency Score | 0.05 |
| Timeouts | 4/100 problems |

## Method 1: Rejection-Sampled SFT

### Data Generation
- 100 problems, 16 samples each
- 98/100 problems had at least one correct answer
- Mean tokens in training data: **1191** (34% reduction from baseline)

### Training
- **Run:** [sft-train-v2](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/15u6i9j1)
- Learning rate: 5e-5
- LoRA rank: 128
- Epochs: 2
- Batch size: 8

### Evaluation
- **Run:** [sft-eval-v2](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/hwkgnpo1)

| Metric | Baseline | SFT | Change |
|--------|----------|-----|--------|
| Accuracy | 91.0% | 83.0% | -8% |
| Mean Tokens | 1805 | 1278 | **-29%** |
| Thinking Tokens | 1308 | 850 | **-35%** |
| Efficiency | 0.05 | 0.06 | **+20%** |
| Timeouts | 4 | 16 | +12 |

### Observations
- SFT reduced token count by 29% and thinking tokens by 35%
- Accuracy dropped 8% (91% → 83%)
- Efficiency improved by 20%
- More timeouts than baseline (16 vs 4) - model may be generating longer on hard problems

## Method 2: Online RL with Efficiency Reward

### Training
- **Run:** [rl-train-v2](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/khmkki1q)
- 100 problems, 1 epoch (4 batches)
- Group size: 8, Groups per batch: 32
- LoRA rank: 128
- Learning rate: 5e-7 (scaled from base 1e-6)

### Evaluation
| Metric | Baseline | RL | Change |
|--------|----------|-----|--------|
| Accuracy | 91.0% | 59.0% | **-32%** |
| Mean Tokens | 1805 | 1302 | -28% |
| Thinking Tokens | 1308 | 622 | **-52%** |
| Efficiency | 0.05 | 0.05 | 0% |
| Timeouts | 4 | 37 | +33 |

### Observations
- RL training was too short (only 4 batches) to be effective
- Many timeouts (37/100) suggest model instability
- Thinking tokens reduced significantly (-52%) but at cost of accuracy
- Would benefit from longer training with more careful hyperparameter tuning

## Key Findings

1. **SFT is the most reliable method** for efficiency training with limited compute
2. **Token reduction achieved**: SFT reduced tokens by 29% while maintaining reasonable accuracy (83%)
3. **RL requires more careful tuning**: Short RL run led to instability and many timeouts
4. **Timeout handling matters**: Counting timeouts as failures gives accurate metrics

## Commands

### Baseline Evaluation
```bash
uv run python -m tinker_cookbook.recipes.math_efficiency.eval \
    model_name="Qwen/Qwen3-8B" \
    num_problems=100 \
    samples_per_problem=4 \
    wandb_name="baseline-eval"
```

### SFT Training Pipeline
```bash
# Step 1: Generate efficient training data
uv run python -m tinker_cookbook.recipes.math_efficiency.generate_efficient_data \
    model_name="Qwen/Qwen3-8B" \
    num_samples=16 \
    num_problems=100 \
    output_path="/tmp/gsm8k_efficient.jsonl"

# Step 2: Train SFT model
uv run python -m tinker_cookbook.recipes.math_efficiency.train_sft \
    model_name="Qwen/Qwen3-8B" \
    data_path="/tmp/gsm8k_efficient.jsonl" \
    learning_rate=5e-5 \
    lora_rank=128 \
    num_epochs=2 \
    log_path="/tmp/math_efficiency_sft" \
    behavior_if_log_dir_exists="delete"
```

### RL Training Pipeline
```bash
uv run python -m tinker_cookbook.recipes.math_efficiency.train_rl \
    model_name="Qwen/Qwen3-8B" \
    num_problems=100 \
    n_epochs=1 \
    group_size=8 \
    groups_per_batch=32 \
    lora_rank=128 \
    log_path="/tmp/math_efficiency_rl" \
    behavior_if_log_dir_exists="delete"
```
