"""
Purpose:
    Clean the raw jsonl results from fact checking tasks that were initially
    returned as invalid JSON.

Inputs:
    Reads all raw jsonl results from the path set as a constant.

Output:
    .parquet files with the following columns:
    - factcheck_analysis_link (str): the unique fact checking link
    - model (str): the name of the model used
    - label (str): the verdict label assigned by the model
    - justification (str, None): explanation provided by the model
    - created (str, None): timestamp when the record was created
    - finish_reason (str): reason why the model stopped generating text
    - num_rag_items (list): number of rag items used in the test
    - used_web_search (bool): indicator if web search was used
    - prompt_tokens (int): number of tokens used in the prompt
    - completion_tokens (int): number of tokens in the model's response
    - total_tokens (int): total token count for the transaction
    - reasoning_tokens (int): number of tokens used in the reasoning

Author: Matthew DeVerna
"""

import json
import os

import pandas as pd

from collections import Counter

from llm_fact.string_helpers import find_closest_string
from llm_fact.parsers import OpenAiFcParser

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))


OUTPUT_DIR = "../../data/cleaned/"
INTERMEDIATE_DIR = OUTPUT_DIR.replace("cleaned", "intermediate")
CLEANED_MESSY_JSON_FILE = os.path.join(
    INTERMEDIATE_DIR, "bad_llm_fc_results_extracted.jsonl"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

POSSIBLE_LABELS = [
    "True",
    "Mostly true",
    "Half true",
    "Mostly false",
    "False",
    "Pants on fire",
    "Not enough information",
]


if __name__ == "__main__":
    # Set up some stuff to hold the records and track bad labels
    records = []
    improper_label = Counter()

    # Load raw results
    with open(CLEANED_MESSY_JSON_FILE, "r") as f:
        data = [json.loads(line) for line in f]

    for response in data:

        # The OpenAiFcParser logic does not work perfectly in this instance as the structure of the response
        # is OpenAI, but the test scenario is from a separate model family. However, this information
        # is stored in the record and we use it to overwrite the returned record.
        parser = OpenAiFcParser(
            input=response,
            input_file_name="fc_results__model_DeepSeek-R1__no_rag.jsonl",  # placeholder fname will be overwritten
        )
        record = parser.parse_fc()
        for key, val in record.items():
            if key in [
                "model",
                "factcheck_analysis_link",
                "num_rag_items",
                "used_web_search",
            ]:
                record[key] = (
                    response[f"{key}_name"] if key == "model" else response[key]
                )
            elif key in ["label", "justification"]:
                pass
            else:
                record[key] = None

        # Mark the input_filename as "messy" to mark responses cleaned by OpenAI
        record["input_filename"] = "messy_json_cleaned_by_openai"
        records.append(record)

    df_good = pd.DataFrame.from_records(records)
    df_good["unknown_label"] = False
    df_good["new_label"] = df_good["label"]

    # Check if any of the labels are not in the possible labels
    mismatched = ~df_good["label"].isin(POSSIBLE_LABELS)
    num_mismatched = sum(mismatched)
    if num_mismatched > 0:
        print(f"\t - Found {num_mismatched:,} unknown labels.")

        # Mark the unknown labels
        df_good["unknown_label"] = mismatched

        print("\t - Using edit distance to find the closest match...")
        # (This works because most mistakes are typically related to capitalization)
        new_labels = []
        for idx, row in df_good.iterrows():
            model = row["model"]
            label = row["label"]
            if label is not None and label not in POSSIBLE_LABELS:
                closest_label = find_closest_string(label, POSSIBLE_LABELS)
                print(f"\t - Replaced <{label}> --with--> <{closest_label}>")
                label = closest_label
                improper_label[model] += 1
            new_labels.append(label)
        df_good["new_label"] = new_labels

    output_filename = "cleaned_factchecking_results_pt2.parquet"
    output_fp = os.path.join(OUTPUT_DIR, output_filename)
    df_good.to_parquet(output_fp)
    print(f"Saved fact-checking results here: {output_fp}\n")
