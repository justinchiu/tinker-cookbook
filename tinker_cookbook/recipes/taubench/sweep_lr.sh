#!/bin/bash
# Parallel learning rate sweep for taubench training
#
# Usage:
#   bash tinker_cookbook/recipes/taubench/sweep_lr.sh
#
# Or from the taubench directory:
#   bash sweep_lr.sh

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
#MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
BASE_LOG_PATH="/tmp/tinker-examples/tau2-lr-sweep"
WANDB_PROJECT="tau2-lr-sweep"  # Set to empty string to disable WandB
DOMAIN="telecom"
BATCH_SIZE=32
GROUP_SIZE=8
NUM_EPOCHS=10

# Learning rates to sweep
LEARNING_RATES=(1e-5 3e-5 5e-5)

# Create base log directory
mkdir -p "$BASE_LOG_PATH"

echo "================================================================================"
echo "TAUBENCH LEARNING RATE SWEEP"
echo "================================================================================"
echo "Learning rates: ${LEARNING_RATES[@]}"
echo "Base log path: $BASE_LOG_PATH"
echo "WandB project: ${WANDB_PROJECT:-None (local only)}"
echo "Model: $MODEL_NAME"
echo "Domain: $DOMAIN"
echo "================================================================================"
echo ""

# Array to store PIDs
pids=()

# Launch all runs in parallel
for lr in "${LEARNING_RATES[@]}"; do
    # Create run-specific log path
    timestamp=$(date +%Y%m%d_%H%M%S)
    run_name="lr_${lr}_${timestamp}"
    log_path="${BASE_LOG_PATH}/${run_name}"

    echo "🚀 Launching: LR=$lr -> $log_path"

    # Build command
    cmd="python -m tinker_cookbook.recipes.taubench.train \
        learning_rate=$lr \
        log_path=$log_path \
        model_name=$MODEL_NAME \
        domain=$DOMAIN \
        batch_size=$BATCH_SIZE \
        group_size=$GROUP_SIZE \
        num_epochs=$NUM_EPOCHS"

    # Add WandB args if configured
    if [ -n "$WANDB_PROJECT" ]; then
        cmd="$cmd wandb_project=$WANDB_PROJECT wandb_name=$run_name"
    fi

    # Launch in background and redirect output to log file
    eval "$cmd" > "${log_path}.stdout.log" 2> "${log_path}.stderr.log" &

    # Store PID
    pid=$!
    pids+=($pid)
    echo "   PID: $pid"

    # Small delay between launches
    sleep 2
done

echo ""
echo "✅ Launched ${#LEARNING_RATES[@]} training runs in parallel"
echo "   PIDs: ${pids[@]}"
echo ""
echo "================================================================================"
echo "Monitoring processes... (Ctrl+C to stop monitoring)"
echo "================================================================================"
echo ""

# Monitor processes
completed=0
failed=0
total=${#pids[@]}

while [ $((completed + failed)) -lt $total ]; do
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        lr=${LEARNING_RATES[$i]}

        # Check if process is still running
        if kill -0 $pid 2>/dev/null; then
            continue
        else
            # Process finished, check exit code
            wait $pid
            exit_code=$?

            if [ $exit_code -eq 0 ]; then
                echo "✅ Completed: LR=$lr (PID $pid)"
                ((completed++))
            else
                echo "❌ Failed: LR=$lr (PID $pid, exit code $exit_code)"
                ((failed++))
            fi

            # Remove from array by marking as done (-1)
            pids[$i]=-1
        fi
    done

    # Remove completed PIDs from array
    pids=(${pids[@]/-1/})

    # Sleep before checking again
    if [ ${#pids[@]} -gt 0 ]; then
        sleep 10
    fi
done

echo ""
echo "================================================================================"
echo "SWEEP COMPLETE"
echo "================================================================================"
echo "Completed: $completed/$total"
echo "Failed: $failed/$total"
echo ""

if [ $failed -eq 0 ]; then
    echo "🎉 All runs completed successfully!"
    exit 0
else
    echo "⚠️  Some runs failed. Check stderr logs in: $BASE_LOG_PATH"
    exit 1
fi
