#!/bin/bash
# Math Efficiency Training Experiment
# Trains Qwen3-8B to solve GSM-8K problems with fewer reasoning tokens
#
# Usage:
#   ./run_experiment.sh              # Run all steps
#   ./run_experiment.sh baseline     # Run only baseline eval
#   ./run_experiment.sh method1      # Run only Method 1 (SFT + eval)
#   ./run_experiment.sh method2      # Run only Method 2 (RL + eval)
#   ./run_experiment.sh method2_answer  # Run RL with answer-hint strategy

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/commands.sh"

run_baseline_eval() { eval_baseline; }

# Function to run Method 1: Rejection-Sampled SFT (includes eval at end)
run_method1() { train_sft; }

# Function to run Method 2: Online RL with Efficiency Reward (includes eval at end)
run_method2() { train_rl; }

run_method2_answer() { train_rl_answer; }

# Function to run Method 2b: RL with answer-hint strategy (IID + answer-hint)
run_method2_answer() {
    echo ""
    echo "========================================"
    echo "Step 3b: Method 2b - RL with Answer Hint Strategy"
    echo "========================================"

    RL_LOG_PATH="${CHECKPOINTS_DIR}/rl_answer_hint"

    uv run python -m tinker_cookbook.recipes.math_efficiency.train_rl \
        model_name="${MODEL_NAME}" \
        num_problems=${NUM_PROBLEMS} \
        group_size=${RL_GROUP_SIZE} \
        groups_per_batch=${RL_GROUPS_PER_BATCH} \
        learning_rate=${RL_LEARNING_RATE} \
        answer_hint_strategy=true \
        lora_rank=${LORA_RANK} \
        max_tokens=${MAX_TOKENS} \
        temperature=${TEMPERATURE} \
        log_path="${RL_LOG_PATH}" \
        wandb_project="${WANDB_PROJECT}" \
        wandb_name="${WANDB_RL_ANSWER_NAME}" \
        eval_num_problems=${NUM_PROBLEMS} \
        eval_samples_per_problem=${SAMPLES_PER_PROBLEM} \
        behavior_if_log_dir_exists="delete"

    # Copy eval results to results dir
    cp "${RL_LOG_PATH}/eval_results.json" "${RESULTS_DIR}/method2_answer_hint_eval.json" 2>/dev/null || true

    echo "Method 2b complete. Checkpoint: ${RL_LOG_PATH}"
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
    method2_answer)
        run_method2_answer
        ;;
    all)
        run_baseline_eval
        run_method1
        run_method2
        run_method2_answer
        print_summary
        ;;
    *)
        echo "Usage: $0 [baseline|method1|method2|method2_answer|all]"
        exit 1
        ;;
esac

echo ""
echo "Done!"
