#!/bin/bash

# Script to run tau2 RL training with proper logging configuration
# tau2 logs go to a separate file, not stdout
#
# Usage:
#   ./run_rl.sh                                    # Run with defaults (all domains)
#   DOMAIN=telecom ./run_rl.sh                     # Train on telecom domain only
#   DOMAIN=airline ./run_rl.sh                     # Train on airline domain only
#   ./run_rl.sh group_size=16 batch_size=128      # Pass training args
#   TAU2_LOG_LEVEL=DEBUG ./run_rl.sh              # Include debug logs
#   MODEL_NAME=meta-llama/Llama-3.1-8B ./run_rl.sh  # Different model
#   MODEL_NAME=Qwen/Qwen3-32B DOMAIN=retail ./run_rl.sh  # Specific model and domain
#   WANDB_PROJECT=my-project ./run_rl.sh          # Custom WandB project
#   CHECKPOINT=tinker://abc123.../weights/000062 ./run_rl.sh  # Continue from checkpoint
#
# WandB Integration (enabled by default if API key is set):
#   export WANDB_API_KEY=your_key_here             # Enable WandB logging
#   WANDB_PROJECT=custom-project ./run_rl.sh       # Override project name
#   WANDB_NAME=custom-run-name ./run_rl.sh         # Override run name
#
# Log levels for TAU2_LOG_LEVEL:
#   DEBUG - Everything including detailed tau2 debug messages (creates large logs!)
#   INFO - Normal operation logs (default)
#   WARNING - Only warnings and errors
#   ERROR - Only errors
#
# Note: tau2.log file is only created when tau2 actually logs something at the specified level

# Default values
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DOMAIN="${DOMAIN:-retail}"  # Default to retail domain
CHECKPOINT="${CHECKPOINT:-}"  # Optional checkpoint to continue from

# Create filesystem-safe model name (replace / with -)
SAFE_MODEL_NAME="${MODEL_NAME//\//-}"

# Include model name and domain in log path for parallel runs
LOG_PATH="${LOG_PATH:-/tmp/tinker-examples/tau2-rl/${SAFE_MODEL_NAME}-${DOMAIN}-$(date +%Y%m%d-%H%M%S)}"
TAU2_LOG_LEVEL="${TAU2_LOG_LEVEL:-INFO}"  # INFO logs everything except DEBUG to file
WANDB_PROJECT="${WANDB_PROJECT:-tau2-rl-experiments}"  # Default WandB project

# Set tau2 log file path (directory will be created by Python script)
TAU2_LOG_FILE="$LOG_PATH/tau2.log"

# Check WandB configuration
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY not set. WandB logging will be disabled."
    echo "   To enable WandB, run: export WANDB_API_KEY=your_key_here"
    WANDB_ARGS=""
else
    # Generate a unique run name with timestamp
    WANDB_NAME="${WANDB_NAME:-tau2-rl-$(date +%Y%m%d-%H%M%S)}"
    WANDB_ARGS="wandb_project=$WANDB_PROJECT wandb_name=$WANDB_NAME"
fi

echo "Starting tau2 RL training..."
echo "=============================="
echo "Model: $MODEL_NAME"
echo "Domain: $DOMAIN"
if [ ! -z "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi
echo "Log path: $LOG_PATH"
echo "tau2 log file: $TAU2_LOG_FILE"
echo "tau2 log level: $TAU2_LOG_LEVEL"
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "WandB: $WANDB_PROJECT / $WANDB_NAME"
fi
echo ""

# Export environment variables for tau2 logging
export TAU2_LOG_LEVEL="$TAU2_LOG_LEVEL"
export TAU2_LOG_FILE="$TAU2_LOG_FILE"

# Build checkpoint argument if provided
CHECKPOINT_ARG=""
if [ ! -z "$CHECKPOINT" ]; then
    CHECKPOINT_ARG="load_checkpoint_path=$CHECKPOINT"
fi

# Run the training with uv
# With TAU2_LOG_FILE set: All tau2 logs (INFO and above) go to file ONLY, not stdout
# Training metrics still go to stdout and $LOG_PATH/metrics.jsonl
uv run python -m tinker_cookbook.recipes.taubench.train \
    model_name="$MODEL_NAME" \
    domain="$DOMAIN" \
    log_path="$LOG_PATH" \
    $CHECKPOINT_ARG \
    $WANDB_ARGS \
    "$@"  # Pass any additional command line arguments

echo ""
echo "Training complete!"
echo "=================="
echo "Training metrics: $LOG_PATH/metrics.jsonl"
if [ -f "$TAU2_LOG_FILE" ]; then
    echo "tau2 logs: $TAU2_LOG_FILE"
    echo ""
    echo "To monitor tau2 logs in real-time, run:"
    echo "  tail -f $TAU2_LOG_FILE"
else
    echo "tau2 logs: $TAU2_LOG_FILE (will be created when tau2 logs at $TAU2_LOG_LEVEL level)"
fi
