#!/bin/bash
# Collect tau2bench rollouts using Claude Opus 4.5
# Usage: ./collect_rollouts_opus.sh [output_dir] [num_trials] [temperature]

set -e

OUTPUT_DIR=${1:-data/tau2_rollouts}
NUM_TRIALS=${2:-16}
MAX_CONCURRENCY=16
TEMPERATURE=${3:-1.0}
AGENT_LLM="claude-opus-4-5-20251101"
USER_LLM="claude-opus-4-5-20251101"

# Convert to absolute path (required for tau2 to save outside default location)
OUTPUT_DIR=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)

echo "=== Collecting tau2bench rollouts (Opus 4.5) ==="
echo "Output dir: $OUTPUT_DIR"
echo "Num trials per task: $NUM_TRIALS"
echo "Max concurrency: $MAX_CONCURRENCY"
echo "Temperature: $TEMPERATURE"
echo "Agent LLM: $AGENT_LLM"
echo "User LLM: $USER_LLM"
echo ""

for DOMAIN in retail airline telecom; do
    echo "=== Running $DOMAIN domain ==="
    OUTPUT_FILE="${OUTPUT_DIR}/${DOMAIN}_${NUM_TRIALS}trials_opus45"
    uv run tau2 run \
        --domain "$DOMAIN" \
        --num-trials "$NUM_TRIALS" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --agent-llm "$AGENT_LLM" \
        --agent-llm-args "{\"temperature\": $TEMPERATURE}" \
        --user-llm "$USER_LLM" \
        --user-llm-args "{\"temperature\": $TEMPERATURE}" \
        --save-to "$OUTPUT_FILE"
    echo ""
done

echo "=== Done! ==="
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.json
