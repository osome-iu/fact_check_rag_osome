#!/bin/bash

# Purpose:
#   Run all Google Gemini tests.
#
# Inputs:
#   None
#
# Output:
#   Each test generates an output in the data/raw/fc_test_results directory.
#
# How to call:
#   bash test_gemini_all.sh
#
# Author: Matthew DeVerna

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }

echo ""
echo "BEGINNING GEMINI FACT CHECKING TESTS"
echo "#########################################"
echo ""

# Set variables for the script and models
script="002_test_gemini_models.py"
models=(
    "gemini-2.0-flash-thinking-exp-01-21"
    "gemini-2.0-pro-exp-02-05"
    "gemini-2.0-flash"
    "gemini-2.0-flash-lite"
    "gemini-1.5-flash-8b-001"
    "gemini-1.5-flash-002"
    "gemini-1.5-pro-002"
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

echo ""
echo "GROUNDED NO RAG"
echo "#########################################"

models=("gemini-2.0-flash-thinking-exp-01-21" "gemini-2.0-flash")

for model in "${models[@]}"; do
    echo "Begin testing model $model with grounded search capability."
    python -u "$script" --model "$model" --grounded
    echo "Test completed for model $model without RAG."
    echo ""
done

echo ""
echo "GROUNDED RAG"
echo "#########################################"

num=6
for model in "${models[@]}"; do
    echo "Begin testing model $model with grounded search capability and $num RAG items."
    python -u "$script" --model "$model" --grounded --with-rag --num-rag-items "$num"
    echo "Test completed for model $model with $num RAG items."
    echo ""
done

echo "Script completed."