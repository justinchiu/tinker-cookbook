#!/bin/bash
# Math Efficiency Training Commands (source this file)
# Usage:
#   source ./commands.sh
#   eval_baseline
#   train_sft
#   train_rl
#   train_rl_answer
#   run_all

set -e

# Configuration
MODEL_NAME="Qwen/Qwen3-8B"
NUM_PROBLEMS=128
SAMPLES_PER_PROBLEM=4
MAX_TOKENS=4096
TEMPERATURE=1.0
LORA_RANK=128

# Wandb configuration
WANDB_PROJECT="math-efficiency-interview"

# Method 1 (SFT) config
SFT_NUM_SAMPLES=16
SFT_LEARNING_RATE=5e-5
SFT_NUM_EPOCHS=2
SFT_BATCH_SIZE=8

# Method 2 (RL) config
RL_GROUP_SIZE=8
RL_GROUPS_PER_BATCH=32
RL_LEARNING_RATE=5e-5
RL_NUM_EPOCHS=15
# 128 problems / 32 groups_per_batch = 4 batches per epoch -> 3 epochs = 12 steps
RL_SAVE_EVERY=10
RL_GROUP_SIZE_X2=16

# Output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/tmp/math_efficiency_experiment_${TIMESTAMP}"
DATA_DIR="${OUTPUT_DIR}/data"
RESULTS_DIR="${OUTPUT_DIR}/results"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints"

WANDB_BASELINE_NAME="baseline-${NUM_PROBLEMS}p-${TIMESTAMP}"
WANDB_SFT_NAME="sft-${NUM_PROBLEMS}p-${SFT_NUM_EPOCHS}ep-lr${SFT_LEARNING_RATE}-${TIMESTAMP}"
WANDB_RL_NAME="rl-${NUM_PROBLEMS}p-${RL_NUM_EPOCHS}ep-${RL_GROUP_SIZE}x${RL_GROUPS_PER_BATCH}-lr${RL_LEARNING_RATE}-${TIMESTAMP}"
WANDB_RL_ANSWER_NAME="rl-${NUM_PROBLEMS}p-${RL_NUM_EPOCHS}ep-${RL_GROUP_SIZE}x${RL_GROUPS_PER_BATCH}-lr${RL_LEARNING_RATE}-answerhint-${TIMESTAMP}"

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}" "${CHECKPOINTS_DIR}"

echo "Configured output dir: ${OUTPUT_DIR}"

eval_baseline() {
    uv run python -m tinker_cookbook.recipes.math_efficiency.eval \
        model_name="${MODEL_NAME}" \
        num_problems=${NUM_PROBLEMS} \
        samples_per_problem=${SAMPLES_PER_PROBLEM} \
        max_tokens=${MAX_TOKENS} \
        temperature=${TEMPERATURE} \
        wandb_project="${WANDB_PROJECT}" \
        wandb_name="${WANDB_BASELINE_NAME}" \
        output_path="${RESULTS_DIR}/baseline_eval.json"
}


train_sft() {
    uv run python -m tinker_cookbook.recipes.math_efficiency.generate_efficient_data \
        model_name="${MODEL_NAME}" \
        num_problems=${NUM_PROBLEMS} \
        num_samples=${SFT_NUM_SAMPLES} \
        max_tokens=${MAX_TOKENS} \
        temperature=${TEMPERATURE} \
        output_path="${DATA_DIR}/gsm8k_efficient.jsonl"

    local sft_log_path="${CHECKPOINTS_DIR}/sft"

    uv run python -m tinker_cookbook.recipes.math_efficiency.train_sft \
        model_name="${MODEL_NAME}" \
        data_path="${DATA_DIR}/gsm8k_efficient.jsonl" \
        learning_rate=${SFT_LEARNING_RATE} \
        num_epochs=${SFT_NUM_EPOCHS} \
        batch_size=${SFT_BATCH_SIZE} \
        lora_rank=${LORA_RANK} \
        max_length=${MAX_TOKENS} \
        log_path="${sft_log_path}" \
        wandb_project="${WANDB_PROJECT}" \
        wandb_name="${WANDB_SFT_NAME}" \
        eval_num_problems=${NUM_PROBLEMS} \
        eval_samples_per_problem=${SAMPLES_PER_PROBLEM} \
        behavior_if_log_dir_exists="delete"

    cp "${sft_log_path}/eval_results.json" "${RESULTS_DIR}/method1_sft_eval.json" 2>/dev/null || true
}

train_rl() {
    local rl_log_path="${CHECKPOINTS_DIR}/rl"

    uv run python -m tinker_cookbook.recipes.math_efficiency.train_rl \
        model_name="${MODEL_NAME}" \
        num_problems=${NUM_PROBLEMS} \
        n_epochs=${RL_NUM_EPOCHS} \
        group_size=${RL_GROUP_SIZE} \
        groups_per_batch=${RL_GROUPS_PER_BATCH} \
        learning_rate=${RL_LEARNING_RATE} \
        lora_rank=${LORA_RANK} \
        max_tokens=${MAX_TOKENS} \
        temperature=${TEMPERATURE} \
        log_path="${rl_log_path}" \
        wandb_project="${WANDB_PROJECT}" \
        wandb_name="${WANDB_RL_NAME}" \
        eval_num_problems=${NUM_PROBLEMS} \
        eval_samples_per_problem=${SAMPLES_PER_PROBLEM} \
        save_every=${RL_SAVE_EVERY} \
        behavior_if_log_dir_exists="delete"

    cp "${rl_log_path}/eval_results.json" "${RESULTS_DIR}/method2_rl_eval.json" 2>/dev/null || true
}

train_rl_answer() {
    local rl_log_path="${CHECKPOINTS_DIR}/rl_answer_hint"

    uv run python -m tinker_cookbook.recipes.math_efficiency.train_rl \
        model_name="${MODEL_NAME}" \
        num_problems=${NUM_PROBLEMS} \
        n_epochs=${RL_NUM_EPOCHS} \
        group_size=${RL_GROUP_SIZE} \
        groups_per_batch=${RL_GROUPS_PER_BATCH} \
        learning_rate=${RL_LEARNING_RATE} \
        answer_hint_strategy=true \
        lora_rank=${LORA_RANK} \
        max_tokens=${MAX_TOKENS} \
        temperature=${TEMPERATURE} \
        log_path="${rl_log_path}" \
        wandb_project="${WANDB_PROJECT}" \
        wandb_name="${WANDB_RL_ANSWER_NAME}" \
        eval_num_problems=${NUM_PROBLEMS} \
        eval_samples_per_problem=${SAMPLES_PER_PROBLEM} \
        save_every=${RL_SAVE_EVERY} \
        behavior_if_log_dir_exists="delete"

    cp "${rl_log_path}/eval_results.json" "${RESULTS_DIR}/method2_answer_hint_eval.json" 2>/dev/null || true
}

train_rl_x2() {
    local rl_log_path="${CHECKPOINTS_DIR}/rl_x2"
    local wandb_name="rl-${NUM_PROBLEMS}p-${RL_NUM_EPOCHS}ep-${RL_GROUP_SIZE_X2}x${RL_GROUPS_PER_BATCH}-lr${RL_LEARNING_RATE}-${TIMESTAMP}"

    uv run python -m tinker_cookbook.recipes.math_efficiency.train_rl \
        model_name="${MODEL_NAME}" \
        num_problems=${NUM_PROBLEMS} \
        n_epochs=${RL_NUM_EPOCHS} \
        group_size=${RL_GROUP_SIZE_X2} \
        groups_per_batch=${RL_GROUPS_PER_BATCH} \
        learning_rate=${RL_LEARNING_RATE} \
        lora_rank=${LORA_RANK} \
        max_tokens=${MAX_TOKENS} \
        temperature=${TEMPERATURE} \
        log_path="${rl_log_path}" \
        wandb_project="${WANDB_PROJECT}" \
        wandb_name="${wandb_name}" \
        eval_num_problems=${NUM_PROBLEMS} \
        eval_samples_per_problem=${SAMPLES_PER_PROBLEM} \
        save_every=${RL_SAVE_EVERY} \
        behavior_if_log_dir_exists="delete"

    cp "${rl_log_path}/eval_results.json" "${RESULTS_DIR}/method2_rl_x2_eval.json" 2>/dev/null || true
}

run_all() {
    eval_baseline
    train_sft
    train_rl
    train_rl_answer
    train_rl_x2
}
