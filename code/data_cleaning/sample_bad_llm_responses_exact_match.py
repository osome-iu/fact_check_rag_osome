"""
Purpose:
    Sample up to 25 example responses per model from the bad LLM fact-checking results,
    match each sample to its cleaned/extracted counterpart using key fields, and produce
    a human-readable comparison file that shows the original messy output alongside the
    cleaned label and justification for easy inspection and analysis.

Input:
    - bad_llm_fc_results.jsonl: JSONL file containing bad LLM responses
    - bad_llm_fc_results_extracted.jsonl: JSONL file containing cleaned/extracted responses

Output:
    - bad_llm_response_samples.txt: Formatted comparison of messy vs cleaned responses

Author: Matthew DeVerna
"""

import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

# Change to script directory
os.chdir(Path(__file__).parent)

# File path constants
BAD_RESULTS_FILE = Path("../../data/intermediate/bad_llm_fc_results.jsonl")
EXTRACTED_RESULTS_FILE = Path(
    "../../data/intermediate/bad_llm_fc_results_extracted.jsonl"
)
OUTPUT_FILE = Path("../../data/intermediate/bad_llm_response_samples_exact_match.txt")


def get_model_name(data):
    """Extract model name from nested structure."""
    model_response = data.get("model_response", {})
    return model_response.get("model") or model_response.get("model_version")


def normalize_model_name(model_name):
    """Normalize model name by removing provider prefix."""
    if not model_name:
        return model_name
    # Remove provider prefixes like 'deepseek-ai/', 'meta-llama/'
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def get_messy_content(data):
    """Extract messy content from model response."""
    try:
        # Try OpenAI/Together AI structure first
        return data["model_response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        try:
            # Try Gemini structure
            return data["model_response"]["candidates"][0]["content"]["parts"][0][
                "text"
            ]
        except (KeyError, IndexError):
            return "ERROR: Could not extract messy content from either structure"


def create_matching_key(data, model_name):
    """Create a key for matching records between files."""
    return (
        data.get("factcheck_analysis_link", ""),
        normalize_model_name(model_name),
        data.get("num_rag_items", ""),
        data.get("used_web_search", ""),
    )


def load_and_sample_bad_results():
    """Load bad results and sample up to 25 per model."""
    model_samples = defaultdict(list)
    line_number = 0

    with open(BAD_RESULTS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line_number += 1
            try:
                data = json.loads(line.strip())
                model_name = get_model_name(data)

                if model_name:
                    normalized_name = normalize_model_name(model_name)
                    model_samples[normalized_name].append(data)
                else:
                    raise ValueError(f"Line {line_number}: Missing model name in JSON")

            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Line {line_number}: {str(e)}", e.doc, e.pos
                )

    # Sample up to 25 records per model
    sampled_data = {}
    for model_name, records in model_samples.items():
        if len(records) <= 25:
            sampled_data[model_name] = records
        else:
            sampled_data[model_name] = random.sample(records, 25)

    return sampled_data


def load_extracted_results():
    """Load extracted results and index by matching key."""
    extracted_index = {}
    line_number = 0

    with open(EXTRACTED_RESULTS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line_number += 1
            try:
                data = json.loads(line.strip())
                model_name = data.get("model_name", "")

                if model_name:
                    matching_key = create_matching_key(data, model_name)
                    extracted_index[matching_key] = data
                else:
                    raise ValueError(f"Line {line_number}: Missing model_name in JSON")

            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Line {line_number}: {str(e)}", e.doc, e.pos
                )

    return extracted_index


def get_extracted_values(extracted_data):
    """Extract label and justification from cleaned data."""
    try:
        parsed = extracted_data["model_response"]["choices"][0]["message"]["parsed"]
        return parsed.get("label", "ERROR: No label"), parsed.get(
            "justification", "ERROR: No justification"
        )
    except (KeyError, IndexError):
        return (
            "ERROR: Could not extract label",
            "ERROR: Could not extract justification",
        )


def extract_label_from_messy_content(messy_content):
    """Extract label from messy JSON content."""
    try:
        # Try to parse as JSON first
        data = json.loads(messy_content)
        return data.get("label", "").strip()
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract label using text matching
        # Remove markdown code blocks if present
        content = re.sub(r"```json\s*|```\s*", "", messy_content)

        # Look for "label": "value" pattern (double quotes)
        match = re.search(r'"label"\s*:\s*"([^"]*)"', content)
        if match:
            return match.group(1).strip()

        # Look for 'label': 'value' pattern (single quotes)
        match = re.search(r"'label'\s*:\s*'([^']*)'", content)
        if match:
            return match.group(1).strip()

        # Look for label: value without quotes
        match = re.search(r"'?\"?label'?\"?\s*:\s*'?\"?([^',\"\}]+)'?\"?", content)
        if match:
            return match.group(1).strip()

        return ""


def labels_match_exactly(messy_content, clean_label):
    """Check if labels match exactly (case-insensitive)."""
    messy_label = extract_label_from_messy_content(messy_content)
    return messy_label.lower() == clean_label.lower()


def write_comparison_file(sampled_data, extracted_index):
    """Write formatted comparison file."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("BAD LLM RESPONSE SAMPLES - MESSY vs CLEANED COMPARISON\n")
        file.write("=" * 80 + "\n\n")

        total_samples = 0
        exact_label_matches = 0
        unmatched_keys = []

        for model_name, records in sampled_data.items():
            file.write(f"MODEL: {model_name}\n")
            file.write("-" * 50 + "\n\n")

            model_exact_matches = 0

            for i, bad_record in enumerate(records, 1):
                total_samples += 1

                # Get original model name for this record
                original_model_name = get_model_name(bad_record)

                # Create matching key (uses normalized model name)
                matching_key = create_matching_key(bad_record, original_model_name)

                # Get messy content
                messy_content = get_messy_content(bad_record)

                file.write(f"SAMPLE {i} ({original_model_name})\n")
                file.write(
                    f"Analysis Link: {bad_record.get('factcheck_analysis_link', 'N/A')}\n"
                )
                file.write(
                    f"RAG Items: {bad_record.get('num_rag_items', 'N/A')}, Web Search: {bad_record.get('used_web_search', 'N/A')}\n"
                )
                file.write(f"Matching Key: {matching_key}\n\n")

                file.write("MESSY ORIGINAL CONTENT:\n")
                file.write(f"{messy_content}\n\n")

                # Try to find matching extracted record
                if matching_key in extracted_index:
                    extracted_record = extracted_index[matching_key]
                    label, justification = get_extracted_values(extracted_record)

                    file.write("CLEANED EXTRACTED VALUES:\n")
                    file.write(f"Label: {label}\n")
                    file.write(f"Justification: {justification}\n\n")

                    # Check for exact label match
                    if labels_match_exactly(messy_content, label):
                        exact_label_matches += 1
                        model_exact_matches += 1
                        file.write("LABEL MATCH: ✓ EXACT MATCH\n\n")
                    else:
                        messy_label = extract_label_from_messy_content(messy_content)
                        file.write(
                            f"LABEL MATCH: ✗ MISMATCH (Messy: '{messy_label}', Clean: '{label}')\n\n"
                        )
                else:
                    file.write("CLEANED EXTRACTED VALUES:\n")
                    file.write("ERROR: No matching extracted record found\n\n")
                    unmatched_keys.append(matching_key)

                file.write("=" * 80 + "\n\n")

            file.write(
                f"Model {model_name} Summary: {model_exact_matches}/{len(records)} exact label matches\n\n"
            )

        # Write debug information
        file.write("DEBUGGING INFORMATION\n")
        file.write("=" * 30 + "\n")
        file.write(f"Total extracted records: {len(extracted_index)}\n")
        file.write("Sample of extracted keys (first 5):\n")
        for i, key in enumerate(list(extracted_index.keys())[:5]):
            file.write(f"  {i+1}: {key}\n")

        file.write("\nSample of unmatched keys (first 5):\n")
        for i, key in enumerate(unmatched_keys[:5]):
            file.write(f"  {i+1}: {key}\n")

        # Write summary statistics
        file.write("\nOVERALL SUMMARY\n")
        file.write("=" * 30 + "\n")
        file.write(f"Total samples: {total_samples}\n")
        file.write(f"Exact label matches: {exact_label_matches}\n")
        file.write(f"Exact match rate: {exact_label_matches/total_samples*100:.1f}%\n")
        file.write(f"Models processed: {len(sampled_data)}\n")


def main():
    """Main function to execute the sampling and comparison process."""
    # Check if input files exist
    if not BAD_RESULTS_FILE.exists():
        print(f"Error: Input file {BAD_RESULTS_FILE} does not exist.")
        return

    if not EXTRACTED_RESULTS_FILE.exists():
        print(f"Error: Input file {EXTRACTED_RESULTS_FILE} does not exist.")
        return

    print("Loading and sampling bad results...")
    sampled_data = load_and_sample_bad_results()

    print("Loading extracted results...")
    extracted_index = load_extracted_results()

    print("Writing comparison file...")
    write_comparison_file(sampled_data, extracted_index)

    print(f"Comparison file saved to: {OUTPUT_FILE}")

    # Print summary
    total_sampled = sum(len(records) for records in sampled_data.values())
    print(f"Sampled {total_sampled} records across {len(sampled_data)} models")


if __name__ == "__main__":
    # Set random seed for reproducible sampling
    random.seed(42)
    main()
