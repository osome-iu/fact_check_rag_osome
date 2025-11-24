#!/bin/bash

# Purpose:
#   Scrape PolitiFact claims/article text, summarize articles, and generate database pipeline.
#
# Inputs:
#   None
#
# Outputs:
#   See individual Python scripts for details.
#
# Prerequisites:
#   - politifact_pkg installed: cd data_collection/politifact_scraper && pip install -e .
#   - OpenAI API key (OPENAI_API_KEY) if running commented-out stages
#
# How to call:
#   bash 00-db-collection-and-generation-pipeline.sh
#
# Author: Matthew DeVerna

set -e  # Exit immediately if a command exits with a non-zero status

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }

echo ""
echo "================================================================================"
echo "  POLITIFACT CLAIM COLLECTION AND DATABASE GENERATION PIPELINE"
echo "================================================================================"
echo ""

#------------------------------------------------------------------------------
# STAGE 0: Scrape PolitiFact Fact-Check Claims
#------------------------------------------------------------------------------
# NOTE: This stage is commented out to preserve reproducibility. Re-scraping
# PolitiFact would produce different results as new fact-checks are continuously
# added to their website. Uncomment this code only if you need to collect fresh
# data for a new analysis.
#
# echo "STAGE 0: Scraping PolitiFact fact-check claims"
# echo "================================================================================"
# cd data_collection/politifact_scraper/scripts
#
# echo "Scraping all PolitiFact fact checks from politifact.com..."
# echo "  Note: This will take some time (thousands of pages to scrape)"
# echo "  Output: ../data/{date}_factchecks.parquet"
# python scrape_politifact_all.py
# echo "Completed: PolitiFact claims scraped successfully"
# echo ""
#
# # Move the output file to data/raw/
# LATEST_FC_FILE=$(ls -t ../data/*_factchecks.parquet 2>/dev/null | head -1)
# cp "$LATEST_FC_FILE" ../../../data/raw/
# echo "Completed: Copied to data/raw/$(basename $LATEST_FC_FILE)"
# echo ""
#
# cd ../../..  # Return to code/ directory

echo "STAGE 0: Skipped (using existing PolitiFact data for reproducibility)"
echo "================================================================================"
echo "Using existing fact-check data from data/raw/"
echo "To scrape fresh data, uncomment Stage 0 in this script"
echo ""

#------------------------------------------------------------------------------
# STAGE 1: Download and Summarize Article Text
#------------------------------------------------------------------------------
# NOTE: This stage is commented out for the same reason as Stage 0. Re-downloading
# articles and generating new summaries would alter the dataset. The original
# summaries are preserved in data/raw/. Uncomment only for new data collection.
#
# echo "STAGE 1: Downloading and summarizing article text"
# echo "================================================================================"
# cd data_collection
#
# echo "Step 1a: Scraping PolitiFact fact-check analysis articles..."
# echo "  Note: Downloads full article text from PolitiFact URLs"
# python 001_scrape_politifact_analysis_text.py
# echo "Completed: Articles saved to data/raw/fc_analysis_text.parquet"
# echo ""
#
# WARNING: This running the below script will incur API costs.
# echo "Step 1b: Summarizing articles using OpenAI..."
# echo "  Note: Requires OPENAI_API_KEY environment variable"
# echo "  Model: GPT-3.5-turbo"
# python 002_summarize_analysis_text.py
# echo "Completed: Summaries saved to data/raw/fc_analysis_text_summaries.jsonl"
# echo ""
#
# cd ..  # Return to code/ directory

echo "STAGE 1: Skipped (using existing article summaries for reproducibility)"
echo "================================================================================"
echo "Using existing summaries from data/raw/fc_analysis_text_summaries.jsonl"
echo "To generate fresh summaries, uncomment Stage 1 in this script"
echo ""

echo "STAGE 2: Cleaning fact-check summaries"
echo "================================================================================"
cd data_cleaning

echo "Executing code/data_cleaning/001_clean_summaries.py"
python 001_clean_summaries.py
echo ""

echo "STAGE 3: Sampling claims for testing"
echo "================================================================================"

echo "Executing code/data_cleaning/002_sample_claims.py"
python 002_sample_claims.py
echo ""

echo "Executing code/data_cleaning/002_subsample_claims.py"
python 002_subsample_claims.py
echo ""

echo "STAGE 4: Building ChromaDB vector database"
echo "================================================================================"
cd ../build_dbs

# NOTE: This script takes ~1.5 hours to build. We therefore exclude it by default allowing users to
# replicate other parts of the pipeline more quickly. If you want to regenerate the database on your own machine,
# uncomment the below code and rerun this script.
# echo "Executing code/build_dbs/build_db.py"
# python build_db.py
# echo ""

echo ""
echo "Skipping database generatino as it takes ~1.5 hours to build (will use existing data for reproducibility)"
echo "To regenerate the ChromaDB database on your machine, uncomment the build_db.py step in this script and rerun it"
echo ""

cd ..  # Return to code/ directory
echo "================================================================================"
echo "  DATABASE PIPELINE COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""
