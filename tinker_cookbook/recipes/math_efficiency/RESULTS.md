# Math Efficiency Training Results

Training Qwen3-8B to solve GSM8K problems with fewer reasoning tokens while maintaining accuracy.

## Experiment Settings
- max_tokens: 4096
- Dataset: 128 fixed problems from GSM8K train set
- Evaluation: 4 samples per problem; accuracy = pass@4 (mean pass_rate)
- Timeout handling: timeouts count as incorrect
- Efficiency: accuracy% / mean_tokens

## Wandb Project
https://wandb.ai/percepta-ai/math-efficiency-interview

## 1. Rejection-Sampled SFT (128 problems)
**Run:** [sft-128p-2ep-lr5e-5-20260130_210152](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/84a638np)

**Training:** LR 5e-5, LoRA rank 128, epochs 2, batch size 8.

**Training data:** 126 problems kept after rejection sampling.

**Evaluation (pass@4 on 128 problems):**

| Metric | Value |
|--------|-------|
| Accuracy | 95.508% |
| Mean Tokens | 1551.0 |
| Mean Thinking Tokens | 1126.2 |
| Efficiency | 0.062 |

## 2. IID RL (group_size=8, groups_per_batch=32)
**Run:** [rl-128p-15ep-8x32-lr5e-5-20260130_205010](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/noq0uw46)

**Training:** 128 problems, 15 epochs, LR 5e-5, LoRA rank 128, save_every 10.

**Learning curve (pass@4 on 128 problems):**

| Step | Accuracy | Mean Tokens | Thinking Tokens | Efficiency |
|------|----------|-------------|----------------|------------|
| 10 | 97.656% | 954.4 | 661.1 | 0.102 |
| 20 | 97.266% | 174.4 | 164.3 | 0.558 |
| 30 | 83.789% | 45.2 | 35.1 | 1.852 |
| 40 | 70.312% | 30.8 | 20.6 | 2.285 |
| 50 | 62.109% | 25.3 | 14.9 | 2.459 |
| 60 | 70.898% | 25.1 | 14.9 | 2.829 |
| final | 70.508% | 25.1 | 14.9 | 2.807 |

## 3. IID RL (group_size=16, groups_per_batch=32)
**Run:** [rl-128p-15ep-16x32-lr5e-5-20260130_221615](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/933qr4xa)

**Training:** 128 problems, 15 epochs, LR 5e-5, LoRA rank 128, save_every 10.

**Learning curve (pass@4 on 128 problems):**

| Step | Accuracy | Mean Tokens | Thinking Tokens | Efficiency |
|------|----------|-------------|----------------|------------|
| 10 | 98.047% | 833.3 | 556.8 | 0.118 |
| 20 | 94.922% | 128.8 | 118.7 | 0.737 |
| 30 | 80.469% | 42.6 | 32.5 | 1.887 |
| 40 | 65.430% | 26.2 | 16.1 | 2.494 |
| 50 | 63.672% | 30.6 | 12.4 | 2.082 |
| 60 | 64.648% | 21.5 | 11.4 | 3.007 |
| final | 64.258% | 21.7 | 11.4 | 2.967 |

## 4. RL + Answer-Hint (IID + answer_hint strategies)
**Run:** [rl-128p-15ep-8x32-lr5e-5-answerhint-20260130_205011](https://wandb.ai/percepta-ai/math-efficiency-interview/runs/sa0lu8vx)

**Training:** 128 problems, 15 epochs, LR 5e-5, LoRA rank 128, save_every 10.

**Learning curve (pass@4 on 128 problems):**

| Step | Accuracy | Mean Tokens | Thinking Tokens | Efficiency |
|------|----------|-------------|----------------|------------|
| 10 | 97.656% | 1177.1 | 865.7 | 0.083 |
| 20 | 98.242% | 624.8 | 425.9 | 0.157 |
| 30 | 97.070% | 196.4 | 175.6 | 0.494 |
| 40 | 68.555% | 58.4 | 32.2 | 1.174 |
| 50 | 82.422% | 44.2 | 34.0 | 1.866 |
| 60 | 75.977% | 33.4 | 23.2 | 2.274 |
| final | 74.805% | 32.9 | 22.8 | 2.277 |
