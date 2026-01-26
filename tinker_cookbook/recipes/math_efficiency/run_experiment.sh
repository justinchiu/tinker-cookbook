#!/bin/bash
# Math Efficiency Training Experiment
# Trains Qwen3-8B to solve GSM-8K problems with fewer reasoning tokens
#
# Usage:
#   ./run_experiment.sh              # Run all steps
#   ./run_experiment.sh baseline     # Run only baseline eval
#   ./run_experiment.sh method1      # Run only Method 1 (SFT + eval)
#   ./run_experiment.sh method2      # Run only Method 2 (RL + eval)

set -e  # Exit on error

# Configuration
MODEL_NAME="Qwen/Qwen3-8B"
NUM_PROBLEMS=10
SAMPLES_PER_PROBLEM=4
MAX_TOKENS=2048
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
RL_BASE_LR=1e-6

# Output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/tmp/math_efficiency_experiment_${TIMESTAMP}"
DATA_DIR="${OUTPUT_DIR}/data"
RESULTS_DIR="${OUTPUT_DIR}/results"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints"

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}" "${CHECKPOINTS_DIR}"

echo "========================================"
echo "Math Efficiency Training Experiment"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Wandb Project: ${WANDB_PROJECT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "========================================"

# Function to run baseline evaluation
run_baseline_eval() {
    echo ""
    echo "========================================"
    echo "Step 1: Baseline Evaluation"
    echo "========================================"

    uv run python -m tinker_cookbook.recipes.math_efficiency.eval \
        model_name="${MODEL_NAME}" \
        num_problems=${NUM_PROBLEMS} \
        samples_per_problem=${SAMPLES_PER_PROBLEM} \
        max_tokens=${MAX_TOKENS} \
        temperature=${TEMPERATURE} \
        wandb_project="${WANDB_PROJECT}" \
        wandb_name="baseline-eval" \
        output_path="${RESULTS_DIR}/baseline_eval.json"

    echo "Baseline results saved to: ${RESULTS_DIR}/baseline_eval.json"
}

# Function to run Method 1: Rejection-Sampled SFT (includes eval at end)
run_method1() {
    echo ""
    echo "========================================"
    echo "Step 2: Method 1 - Rejection-Sampled SFT"
    echo "========================================"

    # Step 2a: Generate efficient training data
    echo ""
    echo "--- Generating efficient training data ---"

    uv run python -m tinker_cookbook.recipes.math_efficiency.generate_efficient_data \
        model_name="${MODEL_NAME}" \
        num_problems=${NUM_PROBLEMS} \
        num_samples=${SFT_NUM_SAMPLES} \
        max_tokens=${MAX_TOKENS} \
        temperature=${TEMPERATURE} \
        output_path="${DATA_DIR}/gsm8k_efficient.jsonl"

    echo "Training data saved to: ${DATA_DIR}/gsm8k_efficient.jsonl"

    # Step 2b: Train SFT on efficient examples (includes eval at end)
    echo ""
    echo "--- Training SFT model (with final evaluation) ---"

    SFT_LOG_PATH="${CHECKPOINTS_DIR}/sft"

    uv run python -m tinker_cookbook.recipes.math_efficiency.train_sft \
        model_name="${MODEL_NAME}" \
        data_path="${DATA_DIR}/gsm8k_efficient.jsonl" \
        learning_rate=${SFT_LEARNING_RATE} \
        num_epochs=${SFT_NUM_EPOCHS} \
        batch_size=${SFT_BATCH_SIZE} \
        lora_rank=${LORA_RANK} \
        max_length=${MAX_TOKENS} \
        log_path="${SFT_LOG_PATH}" \
        wandb_project="${WANDB_PROJECT}" \
        wandb_name="sft-train" \
        eval_num_problems=${NUM_PROBLEMS} \
        eval_samples_per_problem=${SAMPLES_PER_PROBLEM} \
        behavior_if_log_dir_exists="delete"

    # Copy eval results to results dir
    cp "${SFT_LOG_PATH}/eval_results.json" "${RESULTS_DIR}/method1_sft_eval.json" 2>/dev/null || true

    echo "Method 1 complete. Checkpoint: ${SFT_LOG_PATH}"
}

# Function to run Method 2: Online RL with Efficiency Reward (includes eval at end)
run_method2() {
    echo ""
    echo "========================================"
    echo "Step 3: Method 2 - Online RL with Efficiency Reward"
    echo "========================================"

    RL_LOG_PATH="${CHECKPOINTS_DIR}/rl"

    uv run python -m tinker_cookbook.recipes.math_efficiency.train_rl \
        model_name="${MODEL_NAME}" \
        num_problems=${NUM_PROBLEMS} \
        group_size=${RL_GROUP_SIZE} \
        groups_per_batch=${RL_GROUPS_PER_BATCH} \
        base_lr=${RL_BASE_LR} \
        lora_rank=${LORA_RANK} \
        max_tokens=${MAX_TOKENS} \
        temperature=${TEMPERATURE} \
        log_path="${RL_LOG_PATH}" \
        wandb_project="${WANDB_PROJECT}" \
        wandb_name="rl-train" \
        eval_num_problems=${NUM_PROBLEMS} \
        eval_samples_per_problem=${SAMPLES_PER_PROBLEM} \
        behavior_if_log_dir_exists="delete"

    # Copy eval results to results dir
    cp "${RL_LOG_PATH}/eval_results.json" "${RESULTS_DIR}/method2_rl_eval.json" 2>/dev/null || true

    echo "Method 2 complete. Checkpoint: ${RL_LOG_PATH}"
}

# Function to print comparison summary
print_summary() {
    echo ""
    echo "========================================"
    echo "Experiment Complete - Summary"
    echo "========================================"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "Wandb project: ${WANDB_PROJECT}"
    echo ""
    echo "Results files:"
    ls -la "${RESULTS_DIR}/" 2>/dev/null || echo "No results files found"
    echo ""
    echo "View results at: https://wandb.ai/percepta-ai/${WANDB_PROJECT}"
}

# Main execution
case "${1:-all}" in
    baseline)
        run_baseline_eval
        ;;
    method1)
        run_method1
        ;;
    method2)
        run_method2
        ;;
    all)
        run_baseline_eval
        run_method1
        run_method2
        print_summary
        ;;
    *)
        echo "Usage: $0 [baseline|method1|method2|all]"
        exit 1
        ;;
esac

echo ""
echo "Done!"
