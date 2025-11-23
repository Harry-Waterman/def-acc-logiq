#!/bin/bash

# Configuration
# This script benchmarks multiple models sequentially.
# It assumes the API endpoint (e.g., LM Studio, Ollama, or MLC-LLM)
# can either auto-switch models based on the "model" parameter
# OR you will manually switch the model and press [ENTER] when prompted.

API_URL="http://localhost:1234/v1"
SAMPLE_SIZE=400
REPEAT_RUNS=0
AUTO_SWITCH_MODE=true  # Set to true if your server auto-loads models; false to pause for manual switching.
TEMPERATURE=0.7
# Model List
# The user wants to test these specific models.
MODELS=(
   # "qwen/qwen3-4b-2507"
   # "liquid/lfm2-1.2b"
   # "ibm/granite-4-h-tiny"
   # "google/gemma-3-4b"
    "microsoft/phi-4-mini-reasoning"
   # "qwen/qwen3-1.7b"
   # "qwen/qwen3-8b"
)

echo "=================================================="
echo "   Starting Multi-Model Benchmark Suite"
echo "=================================================="
echo "Target Sample Size: $SAMPLE_SIZE"
echo "Models to Test: ${#MODELS[@]}"
echo "--------------------------------------------------"

# Ensure we are in the project root
if [ -d "../../def-acc-logiq" ]; then
    cd ../..
fi

SCRIPT_PATH="def-acc-logiq/evaluation/benchmark_all.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Could not find $SCRIPT_PATH"
    exit 1
fi

for MODEL_NAME in "${MODELS[@]}"; do
    echo ""
    echo "##################################################"
    echo "   PREPARING TO TEST: $MODEL_NAME"
    echo "##################################################"
    
    if [ "$AUTO_SWITCH_MODE" = false ]; then
        echo "ACTION REQUIRED:"
        echo "1. Open your LLM Server (LM Studio / Ollama / MLC)."
        echo "2. Load the model: $MODEL_NAME"
        echo "3. Ensure server is running at $API_URL"
        echo "4. Press [ENTER] to start benchmarking..."
        read -r
    fi
    
    echo ">>> Running Overall Performance Benchmark (Mixed Data)..."
    # Run the mixed benchmark for this model
    python3 "$SCRIPT_PATH" \
        --api-url "$API_URL" \
        --model "$MODEL_NAME" \
        --sample-size "$SAMPLE_SIZE" \
        --repeat-runs "$REPEAT_RUNS" \
        --temperature "$TEMPERATURE" \
        --seed 42
        
    # Renaming the output to include model name for clarity
    # Find the most recent result file
    LATEST_JSON=$(ls -t benchmark_results_seed42_*.json | head -n 1)
    if [ -n "$LATEST_JSON" ]; then
        SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
        NEW_NAME="benchmark_results_${SAFE_MODEL_NAME}_seed42.json"
        mv "$LATEST_JSON" "$NEW_NAME"
        echo "Saved results to: $NEW_NAME"
    fi
    
    echo "--------------------------------------------------"
    echo "   Finished testing: $MODEL_NAME"
    echo "--------------------------------------------------"
done

echo ""
echo "=================================================="
echo "   All Multi-Model Benchmarks Completed"
echo "=================================================="

