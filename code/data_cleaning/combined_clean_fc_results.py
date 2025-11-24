"""
Purpose:
    Combine the two cleaned fact-checking result files into one consistent dataset.

    What this script does
    - Loads the two cleaned inputs:
        - cleaned_factchecking_results_pt1.parquet (primary cleaning pass)
        - cleaned_factchecking_results_pt2.parquet (records re-parsed with gpt-4o-mini)
    - Concatenates them, normalizes model names, fills NaNs for rag item counts, and drops models
        containing "gemini-1.5" (we tested this model but do not report on it as it is outdated).
    - Writes the filtered, combined dataframe to cleaned_factchecking_results_combined.parquet

Inputs:
    - data/cleaned/cleaned_factchecking_results_pt1.parquet
    - data/cleaned/cleaned_factchecking_results_pt2.parquet

Outputs:
    - data/cleaned/cleaned_factchecking_results_combined.parquet

Author: Matthew DeVerna
"""

import os
import warnings

import pandas as pd

from functools import reduce

warnings.simplefilter(action="ignore", category=FutureWarning)

# Ensure we are in the directory where the script is saved
os.chdir(os.path.dirname(os.path.realpath(__file__)))

CLEAN_DIR = "../../data/cleaned/"
PT1_FP = os.path.join(CLEAN_DIR, "cleaned_factchecking_results_pt1.parquet")
PT2_FP = os.path.join(CLEAN_DIR, "cleaned_factchecking_results_pt2.parquet")
OUTPUT_FP = os.path.join(CLEAN_DIR, "cleaned_factchecking_results_combined.parquet")

NUM_CLAIMS_25_PERCENT_SAMPLE = 6153
NUM_CLAIMS_50_PERCENT_SAMPLE = 12306


def get_intersection_set(lists_of_links):
    """
    Get the intersection of a list of lists.

    Parameters
    ----------
    lists_of_links : list of lists
        A list of lists of links.

    Returns
    -------
    set
        A set of links that are in all lists.
    """
    return reduce(lambda a, b: set(a) & set(b), lists_of_links)


if __name__ == "__main__":

    print("Loading data...")
    df_pt1 = pd.read_parquet(PT1_FP)
    df_pt2 = pd.read_parquet(PT2_FP)

    print("Combine the dataframes...")
    df_combined = pd.concat([df_pt1, df_pt2], ignore_index=True)

    print("Unify the model names...")
    df_combined.model = df_combined.model.str.split("/").str[-1]

    print("Filling rag item nans with zeros...")
    df_combined.num_rag_items = df_combined.num_rag_items.fillna(0)

    # We do this because these models are older and not tested in the paper
    print("Drop models with 'gemini-1.5' in the name...")
    df_combined = df_combined[~df_combined.model.str.contains("gemini-1.5")]

    print("Identify links which we have good responses across all test scenarios...")
    small_groups = []
    large_groups = []
    small_sample_links = []
    large_sample_links = []

    grouping_cols = ["model", "num_rag_items", "used_web_search"]
    for group, data in df_combined.groupby(grouping_cols):

        # 25% sample cases
        if data.shape[0] == NUM_CLAIMS_25_PERCENT_SAMPLE:
            small_groups.append(group)
            data = data.dropna(subset=["new_label"])
            small_sample_links.append(data["factcheck_analysis_link"].tolist())

        # 50% sample cases
        elif data.shape[0] == NUM_CLAIMS_50_PERCENT_SAMPLE:
            large_groups.append(group)
            data = data.dropna(subset=["new_label"])
            large_sample_links.append(data["factcheck_analysis_link"].tolist())

        # Throw an error if we have a different number of records (doesn't happen)
        else:
            raise ValueError(f"Unexpected # of records: {group} ({data.shape[0]})")

    # Get the intersection of the links
    small_sample_links = get_intersection_set(small_sample_links)
    large_sample_links = get_intersection_set(large_sample_links)

    # Filter the dataframes to only include the links in the intersection within each group
    print("Filtering the dataframes...")
    reduce_dfs = []
    for group, data in df_combined.groupby(grouping_cols):
        if data.shape[0] == NUM_CLAIMS_25_PERCENT_SAMPLE:
            data = data[data["factcheck_analysis_link"].isin(small_sample_links)]
            reduce_dfs.append(data)

        elif data.shape[0] == NUM_CLAIMS_50_PERCENT_SAMPLE:
            data = data[data["factcheck_analysis_link"].isin(large_sample_links)]
            reduce_dfs.append(data)
        else:
            raise ValueError(f"Unexpected # of records: {group} ({data.shape[0]})")

    # Combine the dataframes
    df_combined = pd.concat(reduce_dfs, ignore_index=True)

    print("Save the combined dataframe...")
    df_combined.to_parquet(OUTPUT_FP, index=False)

    print(f"Combined dataframe saved to {OUTPUT_FP}")
    print("Script completed.")
