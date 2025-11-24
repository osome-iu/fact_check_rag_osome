#!/bin/bash

# Purpose:
#   Run all OpenAI tests.
#
# Inputs:
#   None
#
# Output:
#   Each test generates an output in the data/raw/fc_test_results directory.
#
# How to call:
#   ```
#   bash test_openai_all.sh
#   ```
#
# Author: Matthew DeVerna

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }

# Whether or not to use the test sample (Change to "False" to test the full dataset)
echo ""
echo "BEGINNING OPENAI FACT CHECKING TESTS"
echo "#########################################"

# Set variables for the script and models
script="002_test_openai_models.py"
models=("gpt-4o-mini-2024-07-18" "gpt-4o-2024-11-20")
num_rag_items=(3 6 9)

echo ""
echo "WITHOUT RAG"
echo "#########################################"

# Run the tests for each setting
for model in "${models[@]}"; do
    echo "Begin testing model $model."
    python -u "$script" --model "$model"
    echo "Test completed for model $model."
    echo ""
    echo ""
done


echo ""
echo ""
echo ""
echo "WITH RAG"
echo "####################################"

for model in "${models[@]}"; do
    for num in "${num_rag_items[@]}"; do
        echo "Begin testing model $model with $num RAG items."
        python -u "$script" --model "$model" --with-rag --num-rag-items "$num"
        echo "Test completed for model $model with $num RAG items."
        echo ""
        echo ""
    done
done


echo ""
echo ""
echo ""
echo "GROUNDED NO RAG"
echo "#########################################"

models=("gpt-4o-search-preview" "gpt-4o-mini-search-preview")

for model in "${models[@]}"; do
    echo "Begin testing model $model."
    python -u "$script" --model "$model" --web-search
    echo "Test completed for model $model."
    echo ""
    echo ""
done

echo ""
echo "GROUNDED RAG"
echo "#########################################"

num=6
for model in "${models[@]}"; do
    echo "Begin testing $model with web search capability and $num RAG items."
    python -u "$script" --model "$model" --web-search --with-rag --num-rag-items "$num"
    echo "Test completed for model $model."
    echo ""
    echo ""
done

echo "Script completed."