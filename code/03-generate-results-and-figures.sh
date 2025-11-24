#!/bin/bash

# Purpose:
#   Generate all tables and figures for the manuscript. Creates final research
#   outputs from processed analysis data.
#
# Inputs:
#   None
#
# Outputs:
#   See individual Python scripts for details.
#
#
# How to call:
#   bash 03-generate-results-and-figures.sh
#
# Author: Matthew DeVerna

set -e  # Exit immediately if a command exits with a non-zero status

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }

echo ""
echo "================================================================================"
echo "  GENERATING TABLES AND FIGURES FOR MANUSCRIPT"
echo "================================================================================"
echo ""

echo "STAGE 1: Generating performance tables and reports"
echo "================================================================================"
cd data_analysis

echo "Executing code/data_analysis/generate_performance_tables.py"
python generate_performance_tables.py
echo ""

echo "Executing code/data_analysis/generate_binary_performance_tables.py"
python generate_binary_performance_tables.py
echo ""

echo "Executing code/data_analysis/generate_relative_increase_table.py"
python generate_relative_increase_table.py
echo ""

echo "Executing code/data_analysis/generate_citation_statistics_table.py"
python generate_citation_statistics_table.py
echo ""

echo "Executing code/data_analysis/calculate_faithfulness_agreement.py"
python calculate_faithfulness_agreement.py > ../../reports/faithfulness_agreement_report.txt
echo ""

echo "Executing code/data_cleaning/count_bad_llm_models.py"
cd ../data_cleaning
python count_bad_llm_models.py
echo ""

echo "STAGE 2: Generating performance comparison figures"
echo "================================================================================"
cd ../figure_generation

echo "Executing code/figure_generation/generate_base_model_performance_comparison.py"
python generate_base_model_performance_comparison.py
echo ""

echo "Executing code/figure_generation/generate_base_model_performance_comparison_one_column.py"
python generate_base_model_performance_comparison_one_column.py
echo ""

echo "Executing code/figure_generation/generate_reasoning_web_vs_base_performance.py"
python generate_reasoning_web_vs_base_performance.py
echo ""

echo "Executing code/figure_generation/generate_reasoning_web_vs_base_performance_one_column.py"
python generate_reasoning_web_vs_base_performance_one_column.py
echo ""

echo "Executing code/figure_generation/generate_combined_performance_heatmaps.py"
python generate_combined_performance_heatmaps.py
echo ""

echo "Executing code/figure_generation/generate_6k_vs_12k_performance.py"
python generate_6k_vs_12k_performance.py
echo ""

echo "Executing code/figure_generation/generate_macro_weighted_correlation.py"
python generate_macro_weighted_correlation.py
echo ""

echo "STAGE 3: Generating citation analysis figures"
echo "================================================================================"

echo "Executing code/figure_generation/generate_citation_type_and_top_domains.py"
python generate_citation_type_and_top_domains.py
echo ""

echo "Executing code/figure_generation/generate_citation_type_and_top_domains_one_column.py"
python generate_citation_type_and_top_domains_one_column.py
echo ""

# NOTE: NewsGuard data is proprietary and cannot be shared publicly.
# The following script is commented out because it requires NewsGuard data.
# To uncomment and fully replicate the analysis, users must obtain NewsGuard data separately.
# Then, the code/data_analysis/enrich_web_url_data.py script must be rerun prior to this one.
# echo "Executing code/figure_generation/generate_leaning_newsguard_jointplot.py --dual-marginals"
# python generate_leaning_newsguard_jointplot.py --dual-marginals
# echo ""

# Use Lin et al. (2023) quality scores instead (publicly available data)
echo "Executing code/figure_generation/generate_leaning_lin10k_jointplot.py --dual-marginals"
python generate_leaning_lin10k_jointplot.py --dual-marginals
echo ""

echo "STAGE 4: Generating NEI (Not Enough Information) analysis figure"
echo "================================================================================"

echo "Executing code/figure_generation/generate_nei_plot.py"
python generate_nei_plot.py
echo ""

cd ..  # Return to code/ directory
echo "================================================================================"
echo "  GENERATION COMPLETE - ALL TABLES AND FIGURES CREATED"
echo "================================================================================"
echo ""

