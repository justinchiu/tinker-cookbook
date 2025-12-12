#!/bin/bash

# Script to run tau2 demo rollout with a trained checkpoint
#
# Usage:
#   CHECKPOINT=tinker://abc123.../weights/000062 ./run_demo_rollout_sft.sh
#   CHECKPOINT=tinker://abc123.../weights/final DOMAIN=airline ./run_demo_rollout_sft.sh
#   CHECKPOINT=tinker://abc123.../weights/000062 TASK_ID="[task]" ./run_demo_rollout_sft.sh

# Default values
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DOMAIN="${DOMAIN:-telecom}"
TASK_ID="${TASK_ID:-[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]}"
CHECKPOINT="${CHECKPOINT:-}"
LORA_RANK="${LORA_RANK:-32}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_STEPS="${MAX_STEPS:-10}"

# Check if checkpoint is provided
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT environment variable must be set"
    echo ""
    echo "Usage:"
    echo "  CHECKPOINT=tinker://abc123.../weights/000062 $0"
    echo ""
    echo "Example:"
    echo "  CHECKPOINT=tinker://fe7517d0-2f48-5f5c-8348-287b2b26dc19:train:0/weights/000062 $0"
    exit 1
fi

echo "Starting tau2 demo rollout..."
echo "=============================="
echo "Model: $MODEL_NAME"
echo "Domain: $DOMAIN"
echo "Task ID: $TASK_ID"
echo "Checkpoint: $CHECKPOINT"
echo "LoRA rank: $LORA_RANK"
echo "Max tokens: $MAX_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Max steps: $MAX_STEPS"
echo ""

# Run the demo rollout
uv run python -m tinker_cookbook.recipes.taubench.demo_rollout \
    model_name="$MODEL_NAME" \
    domain="$DOMAIN" \
    task_id="$TASK_ID" \
    checkpoint_path="$CHECKPOINT" \
    lora_rank=$LORA_RANK \
    max_tokens=$MAX_TOKENS \
    temperature=$TEMPERATURE \
    max_steps=$MAX_STEPS

echo ""
echo "Demo rollout completed!"
