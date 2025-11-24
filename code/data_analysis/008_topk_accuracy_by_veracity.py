"""
Purpose:
    Calculate the top-k accuracy of the RAG system for each individual veracity label.

Inputs:
    - retrieval_ranks.parquet: Contains the rank position where each claim's summary was found
    - 2024-10-10_factchecks_cleaned_sampled_0.5_per_year.parquet: Contains veracity labels

Output:
    - topk_accuracy_by_veracity.parquet

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
FACTCHECKS_PATH = os.path.join(
    CLEAN_DATA_DIR, "2024-10-10_factchecks_cleaned_sampled_0.5_per_year.parquet"
)
OUTPUT_PATH = os.path.join(CLEAN_DATA_DIR, "topk_accuracy_by_veracity.parquet")


def load_data():
    """Load retrieval ranks and factcheck data."""
    retrieval_df = pd.read_parquet(RETRIEVAL_RANKS_PATH)
    factchecks_df = pd.read_parquet(
        FACTCHECKS_PATH, columns=["factcheck_analysis_link", "verdict"]
    )
    return retrieval_df, factchecks_df


def calculate_topk_accuracy(merged_df, k_values=[3, 6, 9]):
    """Calculate top-k accuracy for each veracity label."""
    results = []

    for veracity_label, group in merged_df.groupby("verdict"):
        for k in k_values:
            total_claims = len(group)
            claims_found_in_topk = (group["rank"] <= k).sum()
            topk_accuracy = claims_found_in_topk / total_claims

            results.append(
                {
                    "veracity_label": veracity_label,
                    "k_value": k,
                    "total_claims": total_claims,
                    "claims_found_in_topk": claims_found_in_topk,
                    "topk_accuracy": topk_accuracy,
                }
            )

    return pd.DataFrame(results)


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Load data
    retrieval_df, factchecks_df = load_data()

    # Merge retrieval ranks with veracity labels
    merged_df = pd.merge(
        retrieval_df, factchecks_df, on="factcheck_analysis_link", how="inner"
    )

    # Calculate top-k accuracy for each veracity label
    results_df = calculate_topk_accuracy(merged_df)

    # Save results
    results_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"Analyzed {len(results_df)} veracity/k combinations")
