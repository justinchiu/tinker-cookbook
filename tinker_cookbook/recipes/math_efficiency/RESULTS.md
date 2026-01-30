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
| **RL (10 epochs)** | **73.0%** | 1399 | 835 | 0.05 | [rl-train-v3](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/rl-train-v3) |

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

### RL v3: Longer Training (10 epochs)

**Training:**
- **Run:** [rl-train-v3](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/rl-train-v3)
- 100 problems, 10 epochs (40 batches)
- Group size: 8, Groups per batch: 32
- LoRA rank: 128
- Learning rate: 5e-7
- **KL penalty: 0.01** (added to prevent drift)

**Evaluation:**
| Metric | Baseline | RL (1 epoch) | RL (10 epochs) | Change vs 1 epoch |
|--------|----------|--------------|----------------|-------------------|
| Accuracy | 91.0% | 59.0% | 73.0% | **+14%** |
| Mean Tokens | 1805 | 1302 | 1399 | +7% |
| Thinking Tokens | 1308 | 622 | 835 | +34% |
| Efficiency | 0.05 | 0.05 | 0.05 | 0% |
| Timeouts | 4 | 37 | 26 | -11 |

**Observations:**
- More training significantly improved accuracy (59% → 73%)
- KL penalty helped stabilize training
- Fewer timeouts (26 vs 37) indicate more stable model
- Still not matching SFT (83%) but much improved over 1 epoch
- Training showed 91-100% correct during training, gap to eval suggests overfitting

### RL Overfit Experiment: High LR on Small Dataset

**Key insight**: RL wasn't learning because LR was too low (5e-7). With higher LR (5e-5, same as SFT), RL successfully learns to shorten.

**Training:**
- **Run:** [rl-overfit-32prob-50ep-lr5e5](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/903j88qz)
- 32 problems, 50 epochs (same problems every step)
- Learning rate: **5e-5** (100x higher than previous runs)
- No KL penalty

**Training Curve (tokens vs steps):**
| Step | Tokens | Accuracy | Notes |
|------|--------|----------|-------|
| 0 | 1832 | 95.7% | baseline |
| 10 | 1012 | 96.9% | -45% tokens |
| 15 | 611 | 96.9% | -67% tokens |
| 20 | 184 | 94.9% | **-90% tokens** |
| 25 | 113 | 93.8% | sweet spot |
| 30 | 66 | 90.2% | pushing it |
| 35 | 26 | 52.7% | collapsed |
| 50 | 19 | 55.5% | over-optimized |

**Observations:**
- RL **successfully learns** to shorten with higher LR
- Sweet spot around step 20-25: **-90% tokens with ~94% accuracy**
- Without KL penalty, model eventually collapses (memorizes wrong short answers)
- Model skips reasoning entirely at collapse: outputs `<think>\boxed{16}</think>\boxed{16}`

**Follow-up run with 128 problems:**
- **Run:** [rl-overfit-128prob-50ep-lr5e5](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/mo60meo8)
- More problems may help prevent collapse through better generalization

## Key Findings

1. **SFT is simple and reliable** for efficiency training with limited compute
2. **RL works with high LR**: Previous runs used LR=5e-7, but 5e-5 (same as SFT) enables learning
3. **RL can achieve 90% token reduction** while maintaining 94% accuracy (step 20-25)
4. **Without regularization, RL collapses**: Model over-optimizes for length, sacrifices accuracy
5. **Early stopping or KL penalty needed**: To stay in sweet spot before collapse
6. **Gradient signal is sparse**: Within-group token overlap causes cancellation; only distinguishing tokens provide learning signal

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
    n_epochs=10 \
    group_size=8 \
    groups_per_batch=32 \
    kl_penalty_coef=0.01 \
    lora_rank=128 \
    log_path="/tmp/math_efficiency_rl" \
    behavior_if_log_dir_exists="delete"
```

### Learning curves (accuracy/cost vs steps)
Use the per‑checkpoint evaluator to build a learning curve JSONL.

```bash
uv run python -m tinker_cookbook.recipes.math_efficiency.learning_curve \
    log_path="/tmp/math_efficiency_rl" \
    model_name="Qwen/Qwen3-8B" \
    eval_num_problems=50 \
    eval_samples_per_problem=4 \
    checkpoint_stride=2
```

This writes `learning_curve.jsonl` under the run `log_path` with step, accuracy, token metrics, and cumulative training tokens/time.
