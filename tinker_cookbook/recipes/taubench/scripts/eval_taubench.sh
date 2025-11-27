#!/bin/bash
# Collect Tau2 rollouts for evaluator matchups (agent vs user combos) on the retail test split.
# Usage: ./eval_taubench.sh [num_trials] [temperature]

set -euo pipefail

NUM_TRIALS=${1:-8}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-16}
TEMPERATURE=${2:-0}
TASK_SPLIT_NAME=${TASK_SPLIT_NAME:-test}
DOMAIN="retail"

declare -a AGENT_USER_PAIRS=(
    "claude-sonnet-4-5-20250929:claude-sonnet-4-5-20250929"
    "gpt-4.1-2025-04-14:claude-sonnet-4-5-20250929"
    "claude-opus-4-5-20250929:claude-opus-4-5-20250929"
)

echo "=== Collecting Tau2 rollouts for model pairs ==="
echo "Domain: $DOMAIN"
echo "Num trials per task: $NUM_TRIALS"
echo "Task split: $TASK_SPLIT_NAME"
echo "Temperature: $TEMPERATURE"
echo "Pairs:"
printf '  - %s\n' "${AGENT_USER_PAIRS[@]}"
echo ""

for pair in "${AGENT_USER_PAIRS[@]}"; do
    IFS=":" read -r AGENT_LLM USER_LLM <<< "$pair"
    SAFE_PAIR="${AGENT_LLM//[:\/]/_}-${USER_LLM//[:\/]/_}"
    PAIR_DIR="data/eval_test_taubench_${SAFE_PAIR}"
    mkdir -p "$PAIR_DIR"

    echo "=== Domain $DOMAIN :: Agent $AGENT_LLM vs User $USER_LLM ==="
    OUTPUT_FILE="${PAIR_DIR}/retail_${NUM_TRIALS}trials"
    uv run tau2 run \
        --domain "$DOMAIN" \
        --task-split-name "$TASK_SPLIT_NAME" \
        --num-trials "$NUM_TRIALS" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --agent-llm "$AGENT_LLM" \
        --agent-llm-args "{\"temperature\": $TEMPERATURE}" \
        --user-llm "$USER_LLM" \
        --user-llm-args "{\"temperature\": $TEMPERATURE}" \
        --save-to "$OUTPUT_FILE"
    echo "Saved to $OUTPUT_FILE"
    echo ""
done

echo "=== Done! Results saved under data/eval_test_taubench_<agent>_<user> directories ==="
find data -maxdepth 1 -type d -name 'eval_test_taubench_*' -print
