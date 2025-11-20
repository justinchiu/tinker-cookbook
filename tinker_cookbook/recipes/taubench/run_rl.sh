#!/bin/bash

# Script to run tau2 RL training with proper logging configuration
# tau2 logs go to a separate file, not stdout
#
# Usage:
#   ./run_rl.sh                                    # Run with defaults
#   ./run_rl.sh group_size=16 batch_size=128      # Pass training args
#   TAU2_LOG_LEVEL=DEBUG ./run_rl.sh              # Include debug logs
#   MODEL_NAME=meta-llama/Llama-3.1-8B ./run_rl.sh  # Different model
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
LOG_PATH="${LOG_PATH:-/tmp/tinker-examples/tau2-rl/$(date +%Y%m%d-%H%M%S)}"
TAU2_LOG_LEVEL="${TAU2_LOG_LEVEL:-INFO}"  # INFO logs everything except DEBUG to file

# Create log directory
mkdir -p "$LOG_PATH"

# Set tau2 log file path
TAU2_LOG_FILE="$LOG_PATH/tau2.log"

echo "Starting tau2 RL training..."
echo "=============================="
echo "Model: $MODEL_NAME"
echo "Log path: $LOG_PATH"
echo "tau2 log file: $TAU2_LOG_FILE"
echo "tau2 log level: $TAU2_LOG_LEVEL"
echo ""

# Export environment variables for tau2 logging
export TAU2_LOG_LEVEL="$TAU2_LOG_LEVEL"
export TAU2_LOG_FILE="$TAU2_LOG_FILE"

# Run the training with uv
# With TAU2_LOG_FILE set: All tau2 logs (INFO and above) go to file ONLY, not stdout
# Training metrics still go to stdout and $LOG_PATH/metrics.jsonl
uv run python -m tinker_cookbook.recipes.taubench.train \
    model_name="$MODEL_NAME" \
    log_path="$LOG_PATH" \
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