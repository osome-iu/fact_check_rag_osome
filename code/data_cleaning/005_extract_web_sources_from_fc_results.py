"""
Purpose:
    Extract web source information from fact-checking results files.

Inputs:
    Reads all raw jsonl result files that include "_grounded" or "web" in the name.
    Path set as a constant.

Output:
    .parquet files with the following columns:
    - factcheck_analysis_link (str): the unique fact checking link
    - model (str): the name of the model used
    - web_url (str): the web URL extracted from the fact-check content
        None if no web URL is found, and "" if citation objects are provided but no URL is present
    - url_title (str): the title corresponding to the URL
        None if no web URL is found, and "" if citation objects are provided but no URL is present
    - num_rag_items (list): number of rag items used in the test
    - used_web_search (bool): indicator if web search was used (will be True for all)
    - filename (str): the source filename from which data was extracted

    Each row corresponds to one web source, so multiple rows may exist for a single fact-checking link.

Author: Matthew DeVerna
"""

import glob
import json
import os

import pandas as pd

from llm_fact.parsers import OpenAiFcParser, GoogleFcParser

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = "../../data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
FC_RESULTS_DIR = os.path.join(RAW_DATA_DIR, "fc_test_results")
OUTPUT_DIR = os.path.join(DATA_DIR, "cleaned")

if __name__ == "__main__":
    grounded_files = glob.glob(os.path.join(FC_RESULTS_DIR, "*_grounded*"))
    search_files = glob.glob(os.path.join(FC_RESULTS_DIR, "*search*"))
    files = grounded_files + search_files

    records = []
    for file in files:
        print(f"Processing: {file}")
        if "grounded" in file:
            with open(file) as f:
                for line in f:
                    parser = GoogleFcParser(json.loads(line), input_file_name=file)
                    records.extend(parser.parse_web_sources())

        else:
            with open(file) as f:
                for line in f:
                    parser = OpenAiFcParser(json.loads(line), input_file_name=file)
                    records.extend(parser.parse_web_sources())

    # Create DataFrame from records
    df = pd.DataFrame.from_records(records)
    df.fillna({"num_rag_items": 0}, inplace=True)

    # Save DataFrame to Parquet file
    output_file = os.path.join(OUTPUT_DIR, "fc_web_sources.parquet")
    df.to_parquet(output_file, index=False)
