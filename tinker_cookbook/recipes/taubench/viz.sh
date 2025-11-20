#!/bin/bash

# Script to visualize tau2 logs with Streamlit
# Usage: ./viz.sh [log_file_path]

# Default to the latest log file if not specified
if [ -z "$1" ]; then
    # Find the most recent tau2.log file
    LOG_FILE=$(find /tmp/tinker-examples/tau2-rl -name "tau2.log" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)

    if [ -z "$LOG_FILE" ]; then
        echo "No tau2.log files found in /tmp/tinker-examples/tau2-rl"
        echo "Usage: $0 [path/to/tau2.log]"
        exit 1
    fi

    echo "Using most recent log file: $LOG_FILE"
else
    LOG_FILE="$1"
fi

# Check if file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Installing streamlit..."
    pip install streamlit plotly pandas
fi

echo "Starting Streamlit visualization..."
echo "Log file: $LOG_FILE"
echo ""
echo "Opening browser at http://localhost:8501"
echo "Press Ctrl+C to stop"

# Run streamlit with the log file path pre-filled
streamlit run visualize_tau2_logs.py -- "$LOG_FILE"