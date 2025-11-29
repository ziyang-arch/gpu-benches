#!/bin/bash

# Script to run GPU roofline benchmark at constant algorithmic intensity for a specified duration
# Usage: ./run_constant_ai.sh <algorithmic_intensity> <duration_seconds>
# Example: ./run_constant_ai.sh 1.0 60

if [ $# -ne 2 ]; then
    echo "Usage: $0 <algorithmic_intensity> <duration_seconds>"
    echo "  algorithmic_intensity: Target arithmetic intensity in Flop/B (e.g., 0.5, 1.0, 2.0)"
    echo "  duration_seconds: How long to run the benchmark in seconds (e.g., 10, 60, 300)"
    exit 1
fi

ALGORITHMIC_INTENSITY=$1
DURATION=$2

# Check if the executable exists, if not build it
if [ ! -f "./run_constant_ai" ]; then
    echo "Building run_constant_ai..."
    make ./run_constant_ai
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build run_constant_ai"
        exit 1
    fi
fi

# Run the benchmark
./run_constant_ai $ALGORITHMIC_INTENSITY $DURATION

