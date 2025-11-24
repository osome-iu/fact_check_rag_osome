"""
Purpose:
    Clean the raw jsonl results from fact checking tasks.
    Also, extracts poorly formatted JSON responses.

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
from llm_fact.utils import extract_model_name, extract_num_rag_items, used_web_search
from llm_fact.parsers import OpenAiFcParser, TogetherFcParser, GoogleFcParser

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))


RAW_DIR = "../../data/raw/"
DATA_DIR = os.path.join(RAW_DIR, "fc_test_results")
OUTPUT_DIR = RAW_DIR.replace("raw", "cleaned")
INTERMEDIATE_DIR = RAW_DIR.replace("raw", "intermediate")
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
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
    bad_records = []
    couldnt_parse = Counter()
    improper_label = Counter()

    # Load raw results
    files = sorted(os.listdir(DATA_DIR))

    # Set up some file lists
    openai_models = sorted(
        [file for file in files if ("o1" in file or "o3" in file or "gpt-4" in file)]
    )
    together_models = sorted(
        [file for file in files if ("Llama" in file or "DeepSeek" in file)]
    )
    google_models = sorted([file for file in files if "gemini" in file])

    for file in files:
        base_file_name = os.path.basename(file)
        print(f"Working on file: {base_file_name}")

        filepath = os.path.join(DATA_DIR, file)

        # Load the records
        with open(filepath, "r") as f:
            data = [json.loads(line) for line in f]

        # Parse the records
        for record in data:
            if file in openai_models:
                parser = OpenAiFcParser(record, file)
            elif file in together_models:
                parser = TogetherFcParser(record, file)
            elif file in google_models:
                parser = GoogleFcParser(record, file)
            else:
                raise ValueError(f"Can't identify model family for file: {file}")

            try:
                record = parser.parse_fc()
                records.append(record)
            except json.JSONDecodeError:
                # Include what we need to track the test setting
                model_name = extract_model_name(file)
                num_rag_items = extract_num_rag_items(file)
                used_web_search_bool = used_web_search(file)
                record["model_name"] = model_name
                record["num_rag_items"] = num_rag_items
                record["used_web_search"] = used_web_search_bool
                couldnt_parse[model_name] += 1
                bad_records.append(record)

    df_good = pd.DataFrame.from_records(records)

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

    output_filename = "cleaned_factchecking_results_pt1.parquet"
    output_fp = os.path.join(OUTPUT_DIR, output_filename)
    df_good.to_parquet(output_fp)
    print(f"\t - Saved fact-checking results to {output_fp}\n")

    # Save the bad records to a file
    bad_records
    if len(bad_records) > 0:
        bad_filename = "bad_llm_fc_results.jsonl"
        bad_fp = os.path.join(INTERMEDIATE_DIR, bad_filename)
        with open(bad_fp, "w") as f:
            for record in bad_records:
                f.write(json.dumps(record) + "\n")
        print(f"\t - Saved bad records to {bad_fp}\n")
    else:
        print("\t - No bad records found.\n")

    # Print the counts of the models
    print("Counts of responses that we couldn't parse:")
    for model, count in couldnt_parse.items():
        print(f"\t - {model}: {count:,}")
    print("\n")
    # Print the counts of the labels
    print("Counts of improperly formatted labels, fixed with edit distance:")
    for model, count in improper_label.items():
        print(f"\t - {model}: {count:,}")
