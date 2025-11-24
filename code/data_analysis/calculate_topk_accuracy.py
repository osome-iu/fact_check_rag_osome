"""
Purpose:
    Calculate top-k accuracy from retrieval ranks for k values of 3, 6, and 9.
    Top-k accuracy is the proportion of claims where the correct summary
    was found within the top-k retrieved results.

Inputs:
    - retrieval_ranks.parquet: Contains rank positions for each claim

Output:
    - topk_accuracy.parquet: Contains k values and their corresponding accuracies

Author: Matthew DeVerna
"""

import os
import pandas as pd

# Ensure we are in the directory where the script is saved
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# === CONSTANTS ===
DATA_DIR = "../../data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
RETRIEVAL_RANKS_PATH = os.path.join(CLEAN_DATA_DIR, "retrieval_ranks.parquet")
OUTPUT_PATH = os.path.join(CLEAN_DATA_DIR, "topk_accuracy.parquet")


def calculate_topk_accuracy():
    """Calculate top-k accuracy for k values of 3, 6, and 9."""
    # Load retrieval ranks data
    retrieval_df = pd.read_parquet(RETRIEVAL_RANKS_PATH)

    # Define k values to calculate accuracy for
    k_values = [3, 6, 9]

    # Calculate top-k accuracy for each k
    results = []
    total_claims = len(retrieval_df)

    for k in k_values:
        # NOTE: all summaries were found, but it's just good code practice to check
        summary_found_bool = retrieval_df["rank"] > 0
        summary_within_topk_bool = retrieval_df["rank"] <= k
        claims_found_in_topk = (summary_found_bool & summary_within_topk_bool).sum()

        # Calculate accuracy as proportion
        topk_accuracy = claims_found_in_topk / total_claims

        results.append({"k": k, "topk_accuracy": topk_accuracy})

        print(
            f"Top-{k} accuracy: {topk_accuracy:.4f} ({claims_found_in_topk}/{total_claims})"
        )

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    return results_df


if __name__ == "__main__":
    calculate_topk_accuracy()
