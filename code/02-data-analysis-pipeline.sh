#!/bin/bash

# Purpose:
#   Data analysis pipeline. Processes raw fact-checking test results and generates
#   intermediate analysis files needed for tables and figures.
#
# Inputs:
#   None
#
# Outputs:
#   See individual Python scripts for details.
#
# How to call:
#   bash 02-data-analysis-pipeline.sh
#
# Author: Matthew DeVerna

set -e  # Exit immediately if a command exits with a non-zero status

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }

echo ""
echo "================================================================================"
echo "  DATA ANALYSIS PIPELINE"
echo "================================================================================"
echo ""

echo "STAGE 1: Cleaning fact-checking test results"
echo "================================================================================"
cd data_cleaning

echo "Executing code/data_cleaning/004_clean_fc_results_pt1.py"
python 004_clean_fc_results_pt1.py
echo ""

# NOTE: This step is commented out to preserve reproducibility and avoid API costs.
# The extract_unparsable_json_fcs_w_gpt4o_mini.py script uses GPT-4o-mini API to clean
# malformed JSON responses from certain models. The cleaned results are already saved
# in data/intermediate/bad_llm_fc_results_extracted.jsonl. Uncomment only if you need
# to reprocess malformed JSON from new test results.
#
# echo "Executing code/data_cleaning/extract_unparsable_json_fcs_w_gpt4o_mini.py"
# python extract_unparsable_json_fcs_w_gpt4o_mini.py
# echo ""

echo "Skipping GPT4o-mini extraction script (Using existing results for reproducibility)"
echo "To reprocess malformed JSON, uncomment the extract_unparsable_json_fcs_w_gpt4o_mini.py step above"
echo ""

# This script generates an output file for manual review that shows the extraction pipeline worked as intended.
echo "Executing code/data_cleaning/sample_bad_llm_responses_exact_match.py"
python sample_bad_llm_responses_exact_match.py
echo ""

# Now we clean the API results that were fixed using GPT-4o-mini
echo "Executing code/data_cleaning/004_clean_fc_results_pt2_bad_json.py"
python 004_clean_fc_results_pt2_bad_json.py
echo ""

# Combine all cleaned fact-checking results into a single file
echo "Executing code/data_cleaning/combined_clean_fc_results.py"
python combined_clean_fc_results.py
echo ""

echo "STAGE 2: Generating classification reports"
echo "================================================================================"
cd ../data_analysis

echo "Executing code/data_analysis/generate_classification_reports.py"
python generate_classification_reports.py
echo ""

echo "STAGE 3: Processing web source citations"
echo "================================================================================"

# Extract web source URLs from the API responses for OpenAI and Google models (others do not have search capabilities)
echo "Executing code/data_cleaning/005_extract_web_sources_from_fc_results.py"
cd ../data_cleaning
python 005_extract_web_sources_from_fc_results.py
echo ""

# NOTE: This script leverages NewsGuard data, which is proprietary and cannot be shared publicly.
# The script flexibly handles its lack of presence in the repository and the entire pipeline instead
# runs based on the Lin et al (2023) list of domains discussed in the Appendix. A pure replication
# requires that you acquire and save the proper NewsGuard scores with the proper file name, then rerun.
# See the script for details.
echo "Executing code/data_analysis/enrich_web_url_data.py"
cd ../data_analysis
python enrich_web_url_data.py
echo ""

# Calculate proportion of responses that contain web URLs and those where URLs match the original fact-check link
echo "Executing code/data_analysis/analyze_web_url_citations.py"
python analyze_web_url_citations.py
echo ""

echo "STAGE 4: Calculating retrieval analysis metrics"
echo "================================================================================"

# NOTE: This script leverages the ChromaDB vector database built in the 00-db-collection-and-generation-pipeline.sh script.
# By default, the pipeline excludes the generation of the ChromaDB database, as it takes ~1.5 hours to build.
# To fully replicate this step, you must first uncomment that script in the 00-db-collection-and-generation-pipeline.sh
# and rerun it to generate the database.
# echo "Executing code/data_analysis/test_information_retrieval.py"
# python test_information_retrieval.py
# echo ""

echo "Skipping code/data_analysis/test_information_retrieval.py (Using existing database for reproducibility)"
echo "To regenerate the ChromaDB database, uncomment the build_db.py step in the 00-db-collection-and-generation-pipeline.sh script and rerun it"
echo ""

echo "Executing code/data_analysis/calculate_topk_accuracy.py"
python calculate_topk_accuracy.py
echo ""

echo "Executing code/data_analysis/008_topk_accuracy_by_veracity.py"
python 008_topk_accuracy_by_veracity.py
echo ""

echo "STAGE 5: Performing NEI (Not Enough Information) analyses"
echo "================================================================================"

echo "Executing code/data_analysis/006_not_enough_info_analyses.py"
python 006_not_enough_info_analyses.py
echo ""

echo "Executing code/data_analysis/007_nei_performance_by_veracity.py"
python 007_nei_performance_by_veracity.py
echo ""

cd ..  # Return to code/ directory
echo "================================================================================"
echo "  DATA ANALYSIS PIPELINE COMPLETED SUCCESSFULLY"
echo "================================================================================"
