#!/bin/bash

# Purpose:
#   Run tests on together.ai non-reasoning models.
#
# Inputs:
#   None
#
# Output:
#   Each test generates an output in the data/raw/fc_test_results directory.
#
# How to call:
#   bash test_together.sh
#
# Author: Matthew DeVerna

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }

echo ""
echo "BEGINNING TOGETHER FACT CHECKING TESTS"
echo "#########################################"
echo "----------------------------------------"
echo ""

# Set variables for the script and models
script="002_test_togetherai_models.py"

models=(
    "deepseek-ai/DeepSeek-R1"
    "deepseek-ai/DeepSeek-V3"
    "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
)
num_rag_items=(3 6 9)

echo ""
echo "WITHOUT RAG"
echo "#########################################"
for model in "${models[@]}"; do
    echo "Begin testing model $model."
    python -u "$script" --model "$model"
    echo "Test completed for model $model."
    echo ""
done

echo ""
echo "WITH RAG"
echo "#########################################"
for model in "${models[@]}"; do
    for num in "${num_rag_items[@]}"; do
        echo "Begin testing model $model with $num RAG items."
        python -u "$script" --model "$model" --with-rag --num-rag-items "$num"
        echo "Test completed for model $model with $num RAG items."
        echo ""
    done
done