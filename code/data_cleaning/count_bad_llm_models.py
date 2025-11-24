"""
Purpose:
    This script reads a JSONL file containing bad LLM fact-checking results and counts
    the instances of each model based on the "model" key in the nested JSON object.
    The counts are saved to a text file for analysis.

Input:
    - bad_llm_fc_results.jsonl: A JSONL file containing fact-checking results with
      model information in the "model_response.model" field.

Output:
    - bad_llm_model_counts.txt: A text file containing the count of instances for
      each model, sorted by count in descending order.

Author: Matthew DeVerna
"""

import json
import os
from collections import Counter
from pathlib import Path

# Change to script directory
os.chdir(Path(__file__).parent)

# File path constants
INPUT_FILE = Path("../../data/intermediate/bad_llm_fc_results.jsonl")
OUTPUT_FILE = Path("../../data/intermediate/bad_llm_model_counts.txt")


def count_model_instances(jsonl_path, output_path):
    """
    Count instances of each model in the JSONL file and save to text file.

    Args:
        jsonl_path (str): Path to the input JSONL file
        output_path (str): Path to the output text file
    """
    model_counts = Counter()
    line_number = 0

    # Read the JSONL file and count models
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            line_number += 1
            try:
                data = json.loads(line.strip())
                # Extract model from nested structure
                model_response = data.get("model_response", {})
                model_name = model_response.get("model") or model_response.get(
                    "model_version"
                )

                if model_name:
                    model_counts[model_name] += 1
                else:
                    raise ValueError(
                        f"Line {line_number}: Missing 'model_response.model' "
                        "or 'model_response.model_version' field in JSON"
                    )
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Line {line_number}: {str(e)}", e.doc, e.pos
                )

    # Write counts to output file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Model Counts from Bad LLM Fact-Checking Results\n")
        file.write("=" * 50 + "\n\n")

        # Sort by count (descending) then by model name (ascending)
        for model, count in sorted(model_counts.items(), key=lambda x: (-x[1], x[0])):
            file.write(f"{model}: {count}\n")

        file.write(f"\nTotal records processed: {sum(model_counts.values())}\n")
        file.write(f"Unique models found: {len(model_counts)}\n")


def main():
    """Main function to execute the model counting process."""
    # Check if input file exists
    if not INPUT_FILE.exists():
        print(f"Error: Input file {INPUT_FILE} does not exist.")
        return

    # Count model instances and save results
    count_model_instances(INPUT_FILE, OUTPUT_FILE)
    print(f"Model counts saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
