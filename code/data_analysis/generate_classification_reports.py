"""
Purpose:
    Create classification reports for all models/tests given the following scenarios:
    1. Multi-class classification
    2. Binary classification

    We test on two sets of claims:
    - 0.50 sample (~12k claims)
    - 0.25 sample (~6k claims)

    We provide aggregate preformance metrics across all claims as well as
    performance broken down by specific veracity labels and statement year.

    NOTE:
    1. The "Not enough information" label is not included in macro/weighted summary
    scores (e.g., macro-f1) as the f1 score for this label will always be 0.0, since
    this label does not have any true positives.
    2. "Attempted" results drop the "Not enough information" rows before calculating
    performance metrics. This is meant to simulate a system that only attempts to
    fact-check claims it believes it can verify. We do not report on these results
    in the manuscript, but they are included here for completeness.

Inputs:
    Reads cleaned .parquet files from the paths set as constants:
    - FC_VERDICTS_PATH_050: 0.50 sample (~12k claims)
    - FC_VERDICTS_PATH_025: 0.25 sample (~6k claims)

Output:
    Thirty-two files total (sixteen overall + sixteen by-year analysis):

    OVERALL ANALYSIS (16 files - 8 per dataset sample):
    For 0.50 sample:
    - multi_class_all_by_label_0.50.parquet
    - multi_class_all_summary_0.50.parquet
    - multi_class_attempted_by_label_0.50.parquet
    - multi_class_attempted_summary_0.50.parquet
    - binary_class_all_by_label_0.50.parquet
    - binary_class_all_summary_0.50.parquet
    - binary_class_attempted_by_label_0.50.parquet
    - binary_class_attempted_summary_0.50.parquet

    For 0.25 sample:
    - multi_class_all_by_label_0.25.parquet
    - multi_class_all_summary_0.25.parquet
    - multi_class_attempted_by_label_0.25.parquet
    - multi_class_attempted_summary_0.25.parquet
    - binary_class_all_by_label_0.25.parquet
    - binary_class_all_summary_0.25.parquet
    - binary_class_attempted_by_label_0.25.parquet
    - binary_class_attempted_summary_0.25.parquet

    YEAR-BASED ANALYSIS (16 files - 8 per dataset sample):
    Same file structure as above but with '_by_year' suffix. These files contain
    the same metrics but broken out by statement year.

Author: Matthew DeVerna
"""

import os

import numpy as np
import pandas as pd

from collections import Counter

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.multiclass import unique_labels

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))


DATA_DIR = "../../data/cleaned/"
CLEANED_DATA = os.path.join(DATA_DIR, "cleaned_factchecking_results_combined.parquet")
FC_VERDICTS_PATH_050 = os.path.join(
    DATA_DIR, "2024-10-10_factchecks_cleaned_sampled_0.5_per_year.parquet"
)
FC_VERDICTS_PATH_025 = os.path.join(
    DATA_DIR, "2024-10-10_factchecks_cleaned_sampled_0.25_per_year.parquet"
)
OUTPUT_DIR = os.path.join(DATA_DIR, "classification_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VERDICT_ORDER = [
    "True",
    "Mostly true",
    "Half true",
    "Mostly false",
    "False",
    "Pants on fire",
    "Not enough information",
]

TO_BINARY_VERDICT_MAP = {
    "Pants on fire": "False",
    "False": "False",
    "Mostly false": "False",
    "Half true": "True",
    "Mostly true": "True",
    "True": "True",
    "Not enough information": "Not enough information",
}


def generate_performance_results(y_true, y_pred, zero_division=1):
    """
    Calculate precision, recall, f1-score, support, and summary metrics for classification results.
    This function computes the performance metrics for each label, as well as macro and weighted averages.

    Parameters
    ----------
    y_true : list or array-like
        Ground truth labels.
    y_pred : list or array-like
        Predicted labels.
    zero_division : int or str, default=1
        Value to return when division by zero occurs (e.g., when precision or recall is undefined).

    Returns
    -------
    per_label_df : pd.DataFrame
        DataFrame with precision, recall, f1-score, and support for each individual label.

    summary_dict : dict
        Dictionary with flattened summary metrics using keys like:
        'precision-macro-avg', 'recall-weighted-avg', 'accuracy', 'support'
    """
    labels = sorted(unique_labels(y_true, y_pred))

    per_label_rows = []
    precisions = []
    recalls = []
    f1s = []
    supports = []

    # Count true labels for support calculation
    true_counts = Counter(y_true)

    for label in labels:
        y_true_binary = [1 if y == label else 0 for y in y_true]
        y_pred_binary = [1 if y == label else 0 for y in y_pred]

        precision = precision_score(
            y_true_binary, y_pred_binary, zero_division=zero_division
        )
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=zero_division)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=zero_division)
        support = true_counts.get(label, 0)

        per_label_rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support,
            }
        )

        if label.lower() != "not enough information":
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

    # Create per-label DataFrame
    per_label_df = pd.DataFrame(per_label_rows)

    # Calculate macro and weighted averages
    total_support = sum(supports)
    weighted = lambda values: (
        np.average(values, weights=supports) if total_support > 0 else 0.0
    )
    macro = lambda values: np.mean(values) if values else 0.0

    summary_dict = {
        "precision-macro-avg": macro(precisions),
        "recall-macro-avg": macro(recalls),
        "f1-score-macro-avg": macro(f1s),
        "precision-weighted-avg": weighted(precisions),
        "recall-weighted-avg": weighted(recalls),
        "f1-score-weighted-avg": weighted(f1s),
        "accuracy": accuracy_score(y_true, y_pred),
        "support": total_support,
    }

    return per_label_df, summary_dict


def process_dataset(fc_verdicts_path, sample_suffix):
    """
    Process a single dataset and generate overall classification reports.

    Parameters
    ----------
    fc_verdicts_path : str
        Path to the factcheck verdicts parquet file
    sample_suffix : str
        Suffix to add to output files (e.g., "0.50", "0.25")
    """
    print(f"Processing dataset: {fc_verdicts_path} with suffix: {sample_suffix}")

    # Load the data
    ground_truth_fc_df = pd.read_parquet(fc_verdicts_path)
    predicted_fc_df = pd.read_parquet(CLEANED_DATA)

    df = ground_truth_fc_df[["factcheck_analysis_link", "verdict"]].merge(
        predicted_fc_df[
            [
                "factcheck_analysis_link",
                "new_label",  # "new_label" == labels with typos fixed
                "model",
                "num_rag_items",
                "used_web_search",
                "filename",
            ]
        ],
        on="factcheck_analysis_link",
        how="inner",
    )
    df.rename(columns={"new_label": "label"}, inplace=True)
    df.fillna({"num_rag_items": 0}, inplace=True)

    # Some model names look like: deepseek-ai/DeepSeek-V3
    df.model = df.model.str.split("/").str[-1]

    # There are a few models that failed to return results for specific claims.
    # Here we take the intersection of the factcheck_analysis_link across all models
    # and test scenarios to ensure equal comparison.
    groups = df.groupby(["model", "num_rag_items", "used_web_search"])

    sample_threshold = (
        7000  # Controls the size of the group across which we take the intersection
    )
    if sample_suffix == "0.50":
        group_set = {
            name: set(group["factcheck_analysis_link"])
            for name, group in groups
            if len(group) > sample_threshold
        }
    else:
        group_set = {
            name: set(group["factcheck_analysis_link"])
            for name, group in groups
            if len(group) <= sample_threshold
        }

    common_links = set.intersection(*group_set.values()) if group_set else set()
    print("\t - Found common links across all groups:", len(common_links))
    df = df[df["factcheck_analysis_link"].isin(common_links)].reset_index(drop=True)

    for test_scenario in ["multi-class", "binary"]:
        print(f"Calculating results for {test_scenario} classification...")

        if test_scenario == "multi-class":
            df_test = df.copy()
        elif test_scenario == "binary":
            df_test = df.copy()
            df_test["label"] = df_test["label"].map(TO_BINARY_VERDICT_MAP)
            df_test["verdict"] = df_test["verdict"].map(TO_BINARY_VERDICT_MAP)

        label_dfs = []
        summary_dfs = []
        label_dfs_attempted = []
        summary_dfs_attempted = []

        groups_of_interest = ["model", "num_rag_items", "used_web_search"]
        for group_info, data in df_test.groupby(groups_of_interest):

            model = group_info[0]
            num_rag_items = group_info[1]
            used_web_search = group_info[2]
            print(
                f"\t- Working on test: {model}, "
                f"num_rag_items: {num_rag_items}, "
                f"used_web_search: {used_web_search}"
            )

            # Create a temporary dfs for the current group
            temp_data = data.copy()
            temp_data_attempted = temp_data.copy()
            temp_data_attempted = temp_data_attempted[
                temp_data_attempted["label"] != "Not enough information"
            ]

            # Drop missing labels
            temp_data = temp_data.dropna(subset=["label"])
            temp_data_attempted = temp_data_attempted.dropna(subset=["label"])

            # Extract classification report tables
            per_label_df, summary_dict = generate_performance_results(
                temp_data["verdict"], temp_data["label"]
            )
            per_label_df_attempted, summary_dict_attempted = (
                generate_performance_results(
                    temp_data_attempted["verdict"], temp_data_attempted["label"]
                )
            )

            # Add model, num_rag_items, and used_web_search to results
            summary_dict["model"] = model
            summary_dict["num_rag_items"] = num_rag_items
            summary_dict["used_web_search"] = used_web_search
            per_label_df["model"] = model
            per_label_df["num_rag_items"] = num_rag_items
            per_label_df["used_web_search"] = used_web_search

            summary_dict_attempted["model"] = model
            summary_dict_attempted["num_rag_items"] = num_rag_items
            summary_dict_attempted["used_web_search"] = used_web_search
            per_label_df_attempted["model"] = model
            per_label_df_attempted["num_rag_items"] = num_rag_items
            per_label_df_attempted["used_web_search"] = used_web_search

            # Append dfs to the list
            label_dfs.append(per_label_df)
            summary_dfs.append(pd.DataFrame(summary_dict, index=[0]))

            label_dfs_attempted.append(per_label_df_attempted)
            summary_dfs_attempted.append(
                pd.DataFrame(summary_dict_attempted, index=[0])
            )

        # Concatenate all DataFrames
        label_df = pd.concat(label_dfs, ignore_index=True)
        summary_df = pd.concat(summary_dfs, ignore_index=True)

        label_df_attempted = pd.concat(label_dfs_attempted, ignore_index=True)
        summary_df_attempted = pd.concat(summary_dfs_attempted, ignore_index=True)

        # Reorder columns to put grouping variables first
        summary_cols = summary_df.columns
        early_cols = ["model", "num_rag_items", "used_web_search"]

        late_cols = summary_cols[~summary_cols.isin(early_cols)]
        summary_df = pd.concat(
            [summary_df[early_cols], summary_df[late_cols]],
            axis=1,
        )
        summary_df_attempted = pd.concat(
            [summary_df_attempted[early_cols], summary_df_attempted[late_cols]],
            axis=1,
        )

        label_cols = label_df.columns
        early_cols = ["model", "num_rag_items", "used_web_search", "label"]
        late_cols = label_cols[~label_cols.isin(early_cols)]
        label_df = pd.concat(
            [label_df[early_cols], label_df[late_cols]],
            axis=1,
        )
        label_df_attempted = pd.concat(
            [label_df_attempted[early_cols], label_df_attempted[late_cols]],
            axis=1,
        )

        # Set the label columns to be an ordered categorical variable
        cat_type = pd.api.types.CategoricalDtype(categories=VERDICT_ORDER, ordered=True)
        label_df["label"] = label_df["label"].astype(cat_type)
        label_df_attempted["label"] = label_df_attempted["label"].astype(cat_type)

        # Add a "binary" or "multi-class" column
        label_df["test_scenario"] = test_scenario
        label_df_attempted["test_scenario"] = test_scenario

        # Save the results with sample suffix in filename
        # Define output filepaths
        if test_scenario == "multi-class":
            label_path = os.path.join(
                OUTPUT_DIR, f"multi_class_all_by_label_{sample_suffix}.parquet"
            )
            summary_path = os.path.join(
                OUTPUT_DIR, f"multi_class_all_summary_{sample_suffix}.parquet"
            )
            label_attempted_path = os.path.join(
                OUTPUT_DIR, f"multi_class_attempted_by_label_{sample_suffix}.parquet"
            )
            summary_attempted_path = os.path.join(
                OUTPUT_DIR, f"multi_class_attempted_summary_{sample_suffix}.parquet"
            )
        elif test_scenario == "binary":
            label_path = os.path.join(
                OUTPUT_DIR, f"binary_class_all_by_label_{sample_suffix}.parquet"
            )
            summary_path = os.path.join(
                OUTPUT_DIR, f"binary_class_all_summary_{sample_suffix}.parquet"
            )
            label_attempted_path = os.path.join(
                OUTPUT_DIR, f"binary_class_attempted_by_label_{sample_suffix}.parquet"
            )
            summary_attempted_path = os.path.join(
                OUTPUT_DIR, f"binary_class_attempted_summary_{sample_suffix}.parquet"
            )

        # Save the results
        label_df.to_parquet(label_path, index=False)
        summary_df.to_parquet(summary_path, index=False)
        label_df_attempted.to_parquet(label_attempted_path, index=False)
        summary_df_attempted.to_parquet(summary_attempted_path, index=False)

        print(f"Overall results saved to: {OUTPUT_DIR}\n")


def process_dataset_by_year(fc_verdicts_path, sample_suffix):
    """
    Process a single dataset and generate year-based classification reports.
    This function groups the analysis by statement year.

    Parameters
    ----------
    fc_verdicts_path : str
        Path to the factcheck verdicts parquet file
    sample_suffix : str
        Suffix to add to output files (e.g., "0.50", "0.25")
    """
    print(
        f"Processing year-based analysis for: {fc_verdicts_path} with suffix: {sample_suffix}"
    )

    # Load the data
    ground_truth_fc_df = pd.read_parquet(fc_verdicts_path)
    predicted_fc_df = pd.read_parquet(CLEANED_DATA)

    # Extract year from factcheck_date (already datetime)
    ground_truth_fc_df["factcheck_year"] = ground_truth_fc_df["factcheck_date"].dt.year

    df = ground_truth_fc_df[
        ["factcheck_analysis_link", "verdict", "factcheck_year"]
    ].merge(
        predicted_fc_df[
            [
                "factcheck_analysis_link",
                "new_label",  # "new_label" == labels with typos fixed
                "model",
                "num_rag_items",
                "used_web_search",
                "filename",
            ]
        ],
        on="factcheck_analysis_link",
        how="inner",
    )
    df.rename(columns={"new_label": "label"}, inplace=True)
    df.fillna({"num_rag_items": 0}, inplace=True)

    # Some model names look like: deepseek-ai/DeepSeek-V3
    df.model = df.model.str.split("/").str[-1]

    # There are a few models that failed to return results for specific claims.
    # Here we take the intersection of the factcheck_analysis_link across all models
    # and test scenarios to ensure equal comparison.
    groups = df.groupby(["model", "num_rag_items", "used_web_search"])

    sample_threshold = (
        7000  # Controls the size of the group across which we take the intersection
    )
    if sample_suffix == "0.50":
        group_set = {
            name: set(group["factcheck_analysis_link"])
            for name, group in groups
            if len(group) > sample_threshold
        }
    else:
        group_set = {
            name: set(group["factcheck_analysis_link"])
            for name, group in groups
            if len(group) <= sample_threshold
        }

    common_links = set.intersection(*group_set.values()) if group_set else set()
    print("\t - Found common links across all groups:", len(common_links))
    df = df[df["factcheck_analysis_link"].isin(common_links)].reset_index(drop=True)

    for test_scenario in ["multi-class", "binary"]:
        print(f"Calculating year-based results for {test_scenario} classification...")

        if test_scenario == "multi-class":
            df_test = df.copy()
        elif test_scenario == "binary":
            df_test = df.copy()
            df_test["label"] = df_test["label"].map(TO_BINARY_VERDICT_MAP)
            df_test["verdict"] = df_test["verdict"].map(TO_BINARY_VERDICT_MAP)

        label_dfs = []
        summary_dfs = []
        label_dfs_attempted = []
        summary_dfs_attempted = []

        # Group by model, num_rag_items, used_web_search, AND factcheck_year
        groups_of_interest = [
            "model",
            "num_rag_items",
            "used_web_search",
            "factcheck_year",
        ]
        for group_info, data in df_test.groupby(groups_of_interest):

            model = group_info[0]
            num_rag_items = group_info[1]
            used_web_search = group_info[2]
            factcheck_year = group_info[3]
            print(
                f"\t- Working on test: {model}, "
                f"num_rag_items: {num_rag_items}, "
                f"used_web_search: {used_web_search}, "
                f"year: {factcheck_year}"
            )

            # Create a temporary dfs for the current group
            temp_data = data.copy()
            temp_data_attempted = temp_data.copy()
            temp_data_attempted = temp_data_attempted[
                temp_data_attempted["label"] != "Not enough information"
            ]

            # Drop missing labels
            temp_data = temp_data.dropna(subset=["label"])
            temp_data_attempted = temp_data_attempted.dropna(subset=["label"])

            # Skip if no data for this year/model combination
            if len(temp_data) == 0:
                continue

            # Extract classification report tables
            per_label_df, summary_dict = generate_performance_results(
                temp_data["verdict"], temp_data["label"]
            )
            per_label_df_attempted, summary_dict_attempted = (
                generate_performance_results(
                    temp_data_attempted["verdict"], temp_data_attempted["label"]
                )
            )

            # Add model, num_rag_items, used_web_search, and factcheck_year to results
            summary_dict["model"] = model
            summary_dict["num_rag_items"] = num_rag_items
            summary_dict["used_web_search"] = used_web_search
            summary_dict["factcheck_year"] = factcheck_year
            per_label_df["model"] = model
            per_label_df["num_rag_items"] = num_rag_items
            per_label_df["used_web_search"] = used_web_search
            per_label_df["factcheck_year"] = factcheck_year

            summary_dict_attempted["model"] = model
            summary_dict_attempted["num_rag_items"] = num_rag_items
            summary_dict_attempted["used_web_search"] = used_web_search
            summary_dict_attempted["factcheck_year"] = factcheck_year
            per_label_df_attempted["model"] = model
            per_label_df_attempted["num_rag_items"] = num_rag_items
            per_label_df_attempted["used_web_search"] = used_web_search
            per_label_df_attempted["factcheck_year"] = factcheck_year

            # Append dfs to the list
            label_dfs.append(per_label_df)
            summary_dfs.append(pd.DataFrame(summary_dict, index=[0]))

            label_dfs_attempted.append(per_label_df_attempted)
            summary_dfs_attempted.append(
                pd.DataFrame(summary_dict_attempted, index=[0])
            )

        # Concatenate all DataFrames
        label_df = pd.concat(label_dfs, ignore_index=True)
        summary_df = pd.concat(summary_dfs, ignore_index=True)

        label_df_attempted = pd.concat(label_dfs_attempted, ignore_index=True)
        summary_df_attempted = pd.concat(summary_dfs_attempted, ignore_index=True)

        # Reorder columns to put grouping variables first
        summary_cols = summary_df.columns
        early_cols = ["model", "num_rag_items", "used_web_search", "factcheck_year"]

        late_cols = summary_cols[~summary_cols.isin(early_cols)]
        summary_df = pd.concat(
            [summary_df[early_cols], summary_df[late_cols]],
            axis=1,
        )
        summary_df_attempted = pd.concat(
            [summary_df_attempted[early_cols], summary_df_attempted[late_cols]],
            axis=1,
        )

        label_cols = label_df.columns
        early_cols = [
            "model",
            "num_rag_items",
            "used_web_search",
            "factcheck_year",
            "label",
        ]
        late_cols = label_cols[~label_cols.isin(early_cols)]
        label_df = pd.concat(
            [label_df[early_cols], label_df[late_cols]],
            axis=1,
        )
        label_df_attempted = pd.concat(
            [label_df_attempted[early_cols], label_df_attempted[late_cols]],
            axis=1,
        )

        # Set the label columns to be an ordered categorical variable
        cat_type = pd.api.types.CategoricalDtype(categories=VERDICT_ORDER, ordered=True)
        label_df["label"] = label_df["label"].astype(cat_type)
        label_df_attempted["label"] = label_df_attempted["label"].astype(cat_type)

        # Add a "binary" or "multi-class" column
        label_df["test_scenario"] = test_scenario
        label_df_attempted["test_scenario"] = test_scenario

        # Save the results with sample suffix and _by_year in filename
        # Define output filepaths
        if test_scenario == "multi-class":
            label_path = os.path.join(
                OUTPUT_DIR, f"multi_class_all_by_label_{sample_suffix}_by_year.parquet"
            )
            summary_path = os.path.join(
                OUTPUT_DIR, f"multi_class_all_summary_{sample_suffix}_by_year.parquet"
            )
            label_attempted_path = os.path.join(
                OUTPUT_DIR,
                f"multi_class_attempted_by_label_{sample_suffix}_by_year.parquet",
            )
            summary_attempted_path = os.path.join(
                OUTPUT_DIR,
                f"multi_class_attempted_summary_{sample_suffix}_by_year.parquet",
            )
        elif test_scenario == "binary":
            label_path = os.path.join(
                OUTPUT_DIR, f"binary_class_all_by_label_{sample_suffix}_by_year.parquet"
            )
            summary_path = os.path.join(
                OUTPUT_DIR, f"binary_class_all_summary_{sample_suffix}_by_year.parquet"
            )
            label_attempted_path = os.path.join(
                OUTPUT_DIR,
                f"binary_class_attempted_by_label_{sample_suffix}_by_year.parquet",
            )
            summary_attempted_path = os.path.join(
                OUTPUT_DIR,
                f"binary_class_attempted_summary_{sample_suffix}_by_year.parquet",
            )

        # Save the results
        label_df.to_parquet(label_path, index=False)
        summary_df.to_parquet(summary_path, index=False)
        label_df_attempted.to_parquet(label_attempted_path, index=False)
        summary_df_attempted.to_parquet(summary_attempted_path, index=False)

        print(f"Year-based results saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    print("Starting classification report generation for both datasets...")
    print("This script generates both overall and year-based performance analysis.\n")

    # Process both datasets - OVERALL ANALYSIS
    print("=" * 70)
    print("PART 1: OVERALL ANALYSIS")
    print("=" * 70)

    print("\n" + "=" * 60)
    print("PROCESSING 0.50 SAMPLE DATASET (~12k claims) - OVERALL")
    print("=" * 60)
    process_dataset(FC_VERDICTS_PATH_050, "0.50")

    print("\n" + "=" * 60)
    print("PROCESSING 0.25 SAMPLE DATASET (~6k claims) - OVERALL")
    print("=" * 60)
    process_dataset(FC_VERDICTS_PATH_025, "0.25")

    # Process both datasets - YEAR-BASED ANALYSIS
    print("\n" + "=" * 70)
    print("PART 2: YEAR-BASED ANALYSIS")
    print("=" * 70)

    print("\n" + "=" * 60)
    print("PROCESSING 0.50 SAMPLE DATASET (~12k claims) - BY YEAR")
    print("=" * 60)
    process_dataset_by_year(FC_VERDICTS_PATH_050, "0.50")

    print("\n" + "=" * 60)
    print("PROCESSING 0.25 SAMPLE DATASET (~6k claims) - BY YEAR")
    print("=" * 60)
    process_dataset_by_year(FC_VERDICTS_PATH_025, "0.25")

    print("\n" + "=" * 70)
    print("ALL CLASSIFICATION REPORTS COMPLETED SUCCESSFULLY")
    print("Generated 32 files: 16 overall + 16 year-based analysis")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    print("Script finished running.")
