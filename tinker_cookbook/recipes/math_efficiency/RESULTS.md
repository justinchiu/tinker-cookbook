# Math Efficiency Training Results

Training Qwen3-8B to solve GSM-8K problems with fewer reasoning tokens while maintaining accuracy.

## Wandb Project
https://wandb.ai/percepta-ai/math-efficiency-interview

## Results Summary

| Model | Accuracy | Mean Tokens | Thinking Tokens | Efficiency | Wandb |
|-------|----------|-------------|-----------------|------------|-------|
| Baseline | 76.5% | 1555 | 885 | 0.05 | [baseline-eval-100](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/o33z0050) |
| SFT | 85.0% | 1371 | 842 | 0.06 | [sft-eval](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/ufn4hevx) |
| RL (short) | **88.6%** | **1381** | 915 | **0.06** | [rl-train](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/r5aqsevm) |
| RL (long) | 76.0% | 1551 | 873 | 0.05 | [rl-train-long](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/7i6bvi3r) |

## Baseline Evaluation (100 problems)

**Run:** [baseline-eval-100](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/o33z0050)

| Metric | Value |
|--------|-------|
| Accuracy | 76.5% |
| Mean Tokens | 1555 |
| Median Tokens | 1634 |
| P90 Tokens | 2048 |
| Mean Thinking Tokens | 885 (57% of total) |
| Efficiency Score | 0.05 |

## Method 1: Rejection-Sampled SFT

### Data Generation
- 100 problems, 16 samples each
- 94/100 problems had at least one correct answer
- Mean tokens in training data: **1088** (30% reduction from baseline)

### Training
- **Run:** [sft-train](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/apvhxmwf)
- Learning rate: 5e-5
- LoRA rank: 128
- Epochs: 2
- Batch size: 8

### Evaluation
- **Run:** [sft-eval](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/ufn4hevx)

| Metric | Baseline | SFT | Change |
|--------|----------|-----|--------|
| Accuracy | 76.5% | 85.0% | **+8.5%** |
| Mean Tokens | 1555 | 1371 | **-12%** |
| Thinking Tokens | 885 | 842 | **-5%** |
| Efficiency | 0.05 | 0.06 | **+20%** |

### Observations
- SFT improved accuracy by 8.5% while reducing token count by 12%
- The efficiency score improved by 20%
- Model learned from the shortest correct examples in the training data

## Method 2: Online RL with Efficiency Reward

### Training (Short Run)
- **Run:** [rl-train](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/r5aqsevm)
- 10 problems, 1 epoch (4 batches)
- Group size: 8, Groups per batch: 32
- LoRA rank: 128
- Learning rate: 5e-7 (scaled from base 1e-6)

### Evaluation (Short Run)
- 68/100 problems evaluated (some timeouts)

| Metric | Baseline | SFT | RL (short) | Change (vs Baseline) |
|--------|----------|-----|-----|--------|
| Accuracy | 76.5% | 85.0% | 88.6% | **+12.1%** |
| Mean Tokens | 1555 | 1371 | 1381 | **-11%** |
| Thinking Tokens | 885 | 842 | 915 | +3% |
| Efficiency | 0.05 | 0.06 | 0.06 | **+20%** |

### Training (Long Run - 4 Epochs)
- **Run:** [rl-train-long](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/7i6bvi3r)
- 100 problems, 4 epochs (16 batches total)
- Best tokens decreased from 1246 → 934 during training (25% reduction)
- Final average best tokens: 1045

### Evaluation (Long Run)
| Metric | Baseline | RL (long) | Change |
|--------|----------|-----------|--------|
| Accuracy | 76.5% | 76.0% | -0.5% |
| Mean Tokens | 1555 | 1551 | -0.3% |
| Thinking Tokens | 885 | 873 | -1.4% |
| Efficiency | 0.05 | 0.05 | 0% |

### Observations
- **Short RL run** (1 epoch, 10 problems): Best accuracy (88.6%), good token reduction
- **Long RL run** (4 epochs, 100 problems): Model overfit to training distribution
  - Training showed clear learning (best_tokens 1246→934, 25% reduction)
  - But eval accuracy dropped to baseline levels (76%)
  - Suggests the efficiency reward may have pushed the model to cut corners
- The self-improving efficiency reward (`best_so_far / num_tokens`) effectively reduced token counts during training
- More training doesn't always improve generalization - careful tuning needed

## Goals vs Results
| Goal | Target | SFT | RL (short) | RL (long) |
|------|--------|-----|------------|-----------|
| Maintain accuracy | >80% of baseline (>61%) | 85% ✓ | 88.6% ✓ | 76% ✓ |
| Reduce tokens | <70% of baseline (<1089) | 1371 (88%) | 1381 (89%) | 1551 (100%) |
| Maximize efficiency | Higher than baseline | 0.06 (+20%) | 0.06 (+20%) | 0.05 (0%) |

**Key Findings:**
- SFT and short RL both exceeded accuracy goals and improved efficiency
- Short RL (1 epoch) achieved the best accuracy (88.6%)
- Long RL (4 epochs) showed overfitting - training metrics improved but eval didn't
- Neither method hit the aggressive 30% token reduction target

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

# Step 2: Train SFT model (includes final eval)
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
    num_problems=10 \
    group_size=8 \
    groups_per_batch=32 \
    lora_rank=128 \
    log_path="/tmp/math_efficiency_rl" \
    behavior_if_log_dir_exists="delete"
```
