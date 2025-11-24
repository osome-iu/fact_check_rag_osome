#!/usr/bin/env python3
"""
Purpose:
    This script conducts an inter-annotator agreement analysis for faithfulness evaluation
    labels. It compares the accuracy assessments between two independent coders to examine
    consistency in faithfulness judgments, with emphasis on raw agreement statistics and
    label distribution patterns.

Inputs:
    - matts_labels_faithfulness.csv: Manual annotation file from Matt with accuracy labels
    - student_annotator_faithfulness.csv: Manual annotation file from student with accuracy labels

Output:
    - Prints summary results to standard output including:
        - Raw agreement percentage between coders
        - Label distribution and frequency analysis
        - Inter-annotator consistency metrics
        - Krippendorff's alpha coefficient (supplementary measure)

Author:
    Matthew DeVerna
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import sys
import os
import krippendorff

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define file paths
FILE1_PATH = "../../data/manual_annotation/matts_labels_faithfulness.csv"
FILE2_PATH = "../../data/manual_annotation/student_annotator_faithfulness.csv"


def load_and_validate_data(
    file1_path: str, file2_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the two annotation files and perform validation checks.

    Args:
        file1_path: Path to first annotation file
        file2_path: Path to second annotation file

    Returns:
        Tuple of loaded DataFrames
    """

    print("Loading annotation files...")

    # Load the CSV files
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    print(f"Matt's labels: {len(df1)} rows")
    print(f"Student labels: {len(df2)} rows")

    # Assertion check 1: Same number of rows
    assert len(df1) == len(
        df2
    ), f"Files have different number of rows: {len(df1)} vs {len(df2)}"

    # Assertion check 2: Both files have required columns
    required_cols = ["factcheck_analysis_link", "accuracy"]
    for col in required_cols:
        assert col in df1.columns, f"Column '{col}' missing from first file"
        assert col in df2.columns, f"Column '{col}' missing from second file"

    # Sort both dataframes by factcheck_analysis_link to ensure proper alignment
    df1_sorted = df1.sort_values("factcheck_analysis_link").reset_index(drop=True)
    df2_sorted = df2.sort_values("factcheck_analysis_link").reset_index(drop=True)

    # Assertion check 3: factcheck_analysis_link values match exactly
    link_mismatch = df1_sorted["factcheck_analysis_link"].equals(
        df2_sorted["factcheck_analysis_link"]
    )
    assert link_mismatch, "factcheck_analysis_link values do not match between files"

    print("Data validation passed - all assertion checks successful")

    return df1_sorted, df2_sorted


def analyze_accuracy_labels(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the accuracy labels and calculate Krippendorff's alpha.

    Args:
        df1: First annotator's data
        df2: Second annotator's data

    Returns:
        Dictionary with analysis results
    """

    print("\nAnalyzing accuracy labels...")

    # Get accuracy columns
    acc1 = df1["accuracy"].values
    acc2 = df2["accuracy"].values

    # Calculate basic agreement statistics
    exact_matches = np.sum(acc1 == acc2)
    total_items = len(acc1)
    raw_agreement = exact_matches / total_items

    # Get unique labels
    all_labels = np.concatenate([acc1, acc2])
    unique_labels = sorted(set(all_labels))

    # Create label frequency table
    label_freq = {}
    for label in unique_labels:
        count1 = np.sum(acc1 == label)
        count2 = np.sum(acc2 == label)
        label_freq[label] = {
            "annotator1": count1,
            "annotator2": count2,
            "total": count1 + count2,
        }

    # Convert string labels to numeric codes for Krippendorff's alpha
    label_to_code = {label: i for i, label in enumerate(unique_labels)}
    acc1_coded = np.array([label_to_code[label] for label in acc1])
    acc2_coded = np.array([label_to_code[label] for label in acc2])

    # Create data matrix for Krippendorff's alpha (rows = annotators, cols = items)
    data_matrix = np.array([acc1_coded, acc2_coded])

    # Calculate Krippendorff's alpha using the krippendorff package
    alpha = krippendorff.alpha(
        reliability_data=data_matrix, level_of_measurement="nominal"
    )

    return {
        "total_items": total_items,
        "exact_matches": exact_matches,
        "raw_agreement": raw_agreement,
        "unique_labels": unique_labels,
        "label_frequencies": label_freq,
        "krippendorff_alpha": alpha,
    }


def print_results(results: Dict[str, Any]) -> None:
    """
    Print formatted analysis results.

    Args:
        results: Analysis results dictionary
    """

    print("\n" + "=" * 60)
    print("KRIPPENDORFF'S ALPHA ANALYSIS RESULTS")
    print("=" * 60)

    print("\nBasic Statistics:")
    print(f"  Total items analyzed: {results['total_items']}")
    print(f"  Exact matches: {results['exact_matches']}")
    print(
        f"  Raw agreement: {results['raw_agreement']:.4f} ({results['raw_agreement']*100:.2f}%)"
    )

    print(f"\nKrippendorff's Alpha: {results['krippendorff_alpha']:.4f}")
    print(
        "Note: Krippendorff's alpha is misleading here due to extreme label imbalance"
    )
    print(
        "      (one coder uses only one label category, limiting meaningful variance)"
    )

    print("\nLabel Distribution:")
    print(f"{'Label':<20} {'Annotator 1':<12} {'Annotator 2':<12} {'Total':<8}")
    print("-" * 52)

    for label, freq in results["label_frequencies"].items():
        print(
            f"{label:<20} {freq['annotator1']:<12} {freq['annotator2']:<12} {freq['total']:<8}"
        )

    print(f"\nUnique labels found: {results['unique_labels']}")


def main():
    """Main execution function."""

    print("Krippendorff's Alpha Analysis for Faithfulness Annotations")
    print("=" * 60)
    print(f"File 1 (Matt's labels): {FILE1_PATH}")
    print(f"File 2 (Student labels): {FILE2_PATH}")

    # Load and validate data
    df1, df2 = load_and_validate_data(FILE1_PATH, FILE2_PATH)

    # Analyze accuracy labels
    results = analyze_accuracy_labels(df1, df2)

    # Print results
    print_results(results)

    print("\n" + "=" * 60)
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()
