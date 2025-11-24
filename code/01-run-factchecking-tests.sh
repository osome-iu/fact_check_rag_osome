#!/bin/bash

# Purpose:
#   Execute all LLM fact-checking tests. All stages are commented out for reproducibility.
#   Uses existing test results from data/raw/fc_test_results/.
#
# Inputs:
#   None
#
# Outputs:
#   See individual Python scripts for details.
#
# Prerequisites:
#   - ChromaDB database built (run 00-db-collection-and-generation-pipeline.sh first)
#   - API keys (OPENAI_API_KEY, GOOGLE_API_KEY, TOGETHER_API_KEY) if running tests
#   - llm_fact package installed: cd code/package && pip install -e .
#
# How to call:
#   bash 01-run-factchecking-tests.sh
#
# Note:
#   Running all tests is expensive (API costs) and time-consuming (hours to days).
#   Consider running individual scripts in code/factchecking_tests/ instead.
#
# Author: Matthew DeVerna

set -e  # Exit immediately if a command exits with a non-zero status

# Change to the factchecking_tests directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/factchecking_tests" || { echo "Failed to change directory to $SCRIPT_DIR/factchecking_tests"; exit 1; }

echo ""
echo "================================================================================"
echo "  FACT-CHECKING TESTS PIPELINE"
echo "================================================================================"
echo ""
echo "WARNING: This pipeline will execute API calls that incur costs."
echo "         Estimated runtime/cost: Multiple days and thousands of dollars."
echo "         Therefore, we force the user to manually opt into testing. See warnings below."
echo "         Note, also, that perfect replication is unlikely, given how LLMs work."
echo ""

#------------------------------------------------------------------------------
# STAGE 1: OpenAI GPT Model Tests
#------------------------------------------------------------------------------
# NOTE: These stages are commented out to preserve reproducibility. Re-running
# the fact-checking tests would produce different results as:
# 1. LLM outputs are non-deterministic (even with temperature=0, some variation exists)
# 2. Models may be updated by providers over time
# 3. Web search results change constantly
#
# Uncomment these stages to test models with fresh results, but be aware of cost and time implications.
#
# echo "STAGE 1: Running OpenAI GPT model tests"
# echo "================================================================================"
# echo ""
#
# bash test_openai_all.sh
#
# echo "Completed: OpenAI tests finished"
# echo "Output location: data/raw/fc_test_results/gpt-*.jsonl"
# echo ""

echo "STAGE 1: Skipping OpenAI script (using existing results for reproducibility)"
echo "================================================================================"
echo "Using existing results from data/raw/fc_test_results/"
echo "To run fresh tests, uncomment Stage 1 in this script"
echo ""

#------------------------------------------------------------------------------
# STAGE 2: Google Gemini Model Tests
#------------------------------------------------------------------------------
# NOTE: Same reproducibility considerations as Stage 1. Gemini models include
# various experimental and stable versions with different capabilities (thinking,
# grounding, flash/pro variants).
#
# echo "STAGE 2: Running Google Gemini model tests"
# echo "================================================================================"
# echo ""
#
# bash test_gemini_all.sh
#
# echo "Completed: Gemini tests finished"
# echo "Output location: data/raw/fc_test_results/gemini-*.jsonl"
# echo ""

echo "STAGE 2: Skipping Gemini script (using existing results for reproducibility)"
echo "================================================================================"
echo "Using existing results from data/raw/fc_test_results/"
echo "To run fresh tests, uncomment Stage 2 in this script"
echo ""

#------------------------------------------------------------------------------
# STAGE 3: Together AI Model Tests
#------------------------------------------------------------------------------
# NOTE: Same reproducibility considerations as Stages 1-2. Together AI provides
# access to open-source models like DeepSeek and Llama variants.
#
# echo "STAGE 3: Running Together AI model tests (DeepSeek, Llama)"
# echo "================================================================================"
# echo ""
#
# bash test_together.sh
#
# echo "Completed: Together AI tests finished"
# echo "Output location: data/raw/fc_test_results/deepseek-*.jsonl, llama-*.jsonl"
# echo ""

echo "STAGE 3: Skipping Together AI script (using existing results for reproducibility)"
echo "================================================================================"
echo "Using existing results from data/raw/fc_test_results/"
echo "To run fresh tests, uncomment Stage 3 in this script"
echo ""

#------------------------------------------------------------------------------
# STAGE 4: OpenAI Reasoning Model Tests (o1, o3-mini) via Batch API
#------------------------------------------------------------------------------
# NOTE: These stages use OpenAI's Batch API for reasoning models (o1, o3-mini).
# The batch API is more cost-effective and allows processing of many requests.
# These are commented out for the same reproducibility reasons as above.
#
# Step 1: Generate batch files
# echo "STAGE 4a: Generating OpenAI reasoning model batch files"
# echo "================================================================================"
# echo ""
#
# python 003_generate_openai_reasoning_batches.py
#
# echo "Completed: Batch files generated"
# echo "Output location: data/raw/fc_batch_files/"
# echo ""
#
# Step 2: Submit batches and monitor until completion
# echo "STAGE 4b: Submitting and monitoring OpenAI batch jobs"
# echo "================================================================================"
# echo ""
#
# python 003_submit_and_monitor_openai_batches.py
#
# echo "Completed: Batch jobs finished and results saved"
# echo "Output location: data/raw/fc_test_results/o1-*.jsonl, o3-*.jsonl"
# echo ""

echo "STAGE 4: Skipping OpenAI reasoning batch tests (using existing results for reproducibility)"
echo "================================================================================"
echo "Using existing results from data/raw/fc_test_results/"
echo "To run fresh batch tests, uncomment Stage 4 in this script"
echo ""

cd ..  # Return to code/ directory
echo "================================================================================"
echo "  FACT-CHECKING TESTS PIPELINE COMPLETED"
echo "================================================================================"
echo ""
