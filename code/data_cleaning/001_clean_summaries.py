"""
Purpose: 
    1. Clean the OpenAI summaries of PolitiFact fact checking articles.
    2. Create a matching fact check frame with the final links for which we have summaries.

Inputs:
    None. Input is read from paths set as constants.

Output:
    .parquet file containing the cleaned OpenAI summaries of the PolitiFact fact checking articles.
    Columns:
        - summary (str): the content of the message.
        - created (int): the creation timestamp.
        - model (str): the model used.
        - completion_tokens (int): the number of completion tokens used.
        - prompt_tokens (int): the number of prompt tokens used.
        - total_tokens (int): the total number of tokens used.
        - factcheck_analysis_link (str or None): the link to the fact-check analysis.
        - finish_reason (str): the reason for finishing the response.

Author: Matthew DeVerna
"""

import json
import os
import pandas as pd

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from llm_fact.parsers import parse_summary

# Set paths and create output directory
DATA_DIR = "../../data/raw"
FC_FILENAME = "2024-10-10_factchecks.parquet"
SUMMARIES_FILE = os.path.join(DATA_DIR, "fc_analysis_text_summaries.jsonl")
FC_FRAME_FILE = os.path.join(DATA_DIR, FC_FILENAME)
OUTPUT_DIR = "../../data/cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# This is used to clean the text before passing to OpenAI
VERDICT_MAP = {
    "true": "True",
    "mostly-true": "Mostly true",
    "half-true": "Half true",
    "mostly-false": "Mostly false",
    "false": "False",
    "tom_ruling_pof": "Pants on fire",
    "full-flop": "Full flop",
    "half-flip": "Half flip",
    "no-flip": "No flip",
}


if __name__ == "__main__":
    summaries = []
    print("Loading and cleaning summaries...")
    with open(SUMMARIES_FILE, "r") as f:
        for line in f:
            response = json.loads(line)
            flat_response = parse_summary(response)
            summaries.append(flat_response)
    summaries = pd.DataFrame(summaries)

    print("Removing bad rows...")
    # Remove duplicates, summaries that are empty or contain the default message
    initial_count = len(summaries)

    # Create masks for each condition
    duplicate_mask = summaries.duplicated()
    empty_summary_mask = summaries["summary"] == ""
    default_message_mask = summaries["summary"].str.contains(
        "No article text provided."
    )
    short_summary_mask = summaries["summary"].str.len() < 200

    # Print the number of rows that meet each condition
    print(f"\t- Duplicates: {duplicate_mask.sum()}")
    print(f"\t- Empty summaries: {empty_summary_mask.sum()}")
    print(f"\t- Default message: {default_message_mask.sum()}")
    print(f"\t- Short summaries: {short_summary_mask.sum()}")

    # Combine all masks to remove rows that meet any of the conditions
    combined_mask = (
        duplicate_mask | empty_summary_mask | default_message_mask | short_summary_mask
    )
    rows_to_remove = combined_mask.sum()
    summaries = summaries[~combined_mask]
    print(f"\t- Total removed: {rows_to_remove}")

    final_count = len(summaries)
    print(f"Total rows removed: {initial_count - final_count}")
    summaries.reset_index(drop=True, inplace=True)

    print("Saving the cleaned summaries...")
    output_fp = os.path.join(OUTPUT_DIR, "factcheck_summaries.parquet")
    summaries.to_parquet(output_fp, index=False)
    print(f"Saved: {output_fp}")

    print("Creating a clean fact check frame with only the summarized links...")
    # Load the fact check frame to match the links
    factchecks_df = pd.read_parquet(FC_FRAME_FILE)
    factchecks_df = factchecks_df[
        factchecks_df["factcheck_analysis_link"].isin(
            summaries["factcheck_analysis_link"]
        )
    ]
    factchecks_df["verdict"] = factchecks_df["verdict"].map(VERDICT_MAP)

    factchecks_df.drop_duplicates(subset="factcheck_analysis_link", inplace=True)
    factchecks_df.reset_index(drop=True, inplace=True)
    cleaned_fc_filename = FC_FILENAME.replace(".parquet", "_cleaned.parquet")
    output_fp = os.path.join(OUTPUT_DIR, cleaned_fc_filename)
    factchecks_df.to_parquet(output_fp, index=False)
    print(f"Saved: {output_fp}")
