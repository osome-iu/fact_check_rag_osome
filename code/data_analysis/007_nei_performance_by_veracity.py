"""
Purpose:
    Analyze the use of the "not enough information" (NEI) label by claim veracity label.
    For each ground truth veracity label (True, Mostly True, Half True, Mostly False, False, Pants on Fire),
    calculate binary NEI performance metrics including precision, recall, F1, and accuracy.

Notes:
    - This script analyzes the NEI label using confusion matrix quadrants for each veracity label:
        - True Positive (TP): Model correctly predicts NEI when supporting evidence is absent.
        - True Negative (TN): Model correctly predicts not NEI when supporting evidence is present.
        - False Positive (FP): Model predicts NEI when supporting evidence is present (overuse).
        - False Negative (FN): Model fails to predict NEI when supporting evidence is absent (underuse).
    - Results are stratified by model, retrieval settings, web search usage, and veracity label.

Inputs:
    See the constants below for the paths to the input files.

Outputs:
    nei_performance_metrics_by_veracity.parquet
        - model: Name of the model evaluated.
        - num_rag_items: Number of RAG items used for retrieval.
        - used_web_search: Whether web search was used (bool).
        - veracity_label: Ground truth veracity label (True, Mostly True, etc.).
        - precision: Precision for NEI as a binary classification for this veracity label.
        - recall: Recall for NEI as a binary classification for this veracity label.
        - f1_score: F1 score for NEI as a binary classification for this veracity label.
        - accuracy: Accuracy for NEI as a binary classification for this veracity label.
    nei_confusion_summary_by_veracity.parquet
        - model: Name of the model evaluated.
        - num_rag_items: Number of RAG items used for retrieval.
        - used_web_search: Whether web search was used (bool).
        - veracity_label: Ground truth veracity label (True, Mostly True, etc.).
        - true_positive: Count of claims where the model correctly predicted NEI when it should have.
        - false_positive: Count of claims where the model predicted NEI when it should not have (overuse).
        - true_negative: Count of claims where the model correctly predicted not NEI when it should not have.
        - false_negative: Count of claims where the model failed to predict NEI when it should have (underuse).
    nei_results_by_veracity.parquet
        - model: Name of the model evaluated.
        - num_rag_items: Number of RAG items used for retrieval.
        - used_web_search: Whether web search was used (bool).
        - veracity_label: Ground truth veracity label (True, Mostly True, etc.).
        - total_claims: Total number of claims for this veracity label.
        - nei_predictions: Number of claims predicted as NEI.
        - nei_proportion: Proportion of claims predicted as NEI.
        - summary_in_rag_results: Number of claims where correct summary was found in RAG results.
        - summary_not_in_rag_results: Number of claims where correct summary was not found in RAG results.

Author: Matthew DeVerna
"""

import os
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# Ensure we are in the dir where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === CONSTANTS ===
DATA_DIR = "../../data/cleaned"
RETRIEVAL_PATH = os.path.join(DATA_DIR, "retrieval_ranks.parquet")
FACTCHECKS_PATH = os.path.join(
    DATA_DIR, "2024-10-10_factchecks_cleaned_sampled_0.5_per_year.parquet"
)
RESULTS_PATH = os.path.join(DATA_DIR, "cleaned_factchecking_results_combined.parquet")
NEI_PERFORMANCE_METRICS_BY_VERACITY_PATH = os.path.join(
    DATA_DIR, "nei_performance_metrics_by_veracity.parquet"
)
NEI_CONFUSION_SUMMARY_BY_VERACITY_PATH = os.path.join(
    DATA_DIR, "nei_confusion_summary_by_veracity.parquet"
)
NEI_RESULTS_BY_VERACITY_PATH = os.path.join(DATA_DIR, "nei_results_by_veracity.parquet")


# === FUNCTIONS ===
def analyze_nei_behavior_by_veracity(temp_df, rank_cutoff="in_top_9"):
    """
    Returns results of a binary classification analysis and a confusion matrix for
    the "Not enough information" (NEI) label for each veracity label.

    Parameters
    ----------
    temp_df : pd.DataFrame
        DataFrame with model predictions and retrieval coverage indicators.
    rank_cutoff : str, default="in_top_9"
        Column name (e.g., 'in_top_3', 'in_top_6', 'in_top_9') used to determine if
        the correct document was retrieved.

    Returns
    -------
    results : list
        List of dictionaries containing metrics and confusion matrix data for each veracity label.
    """
    assert rank_cutoff in {"in_top_3", "in_top_6", "in_top_9"}, "Invalid rank_cutoff"

    results = []

    # Group by veracity label and calculate metrics for each
    for veracity_label, veracity_df in temp_df.groupby("verdict"):
        # Ground truth: should the model say NEI?
        # If correct summary not in top-k, model *should* say NEI
        y_should_be_nei = (~veracity_df[rank_cutoff]).astype(int)

        # Model prediction: did the model say NEI?
        y_predicted_nei = (veracity_df["label"] == "Not enough information").astype(int)

        # Binary classification metrics
        precision = precision_score(y_should_be_nei, y_predicted_nei, zero_division=0)
        recall = recall_score(y_should_be_nei, y_predicted_nei, zero_division=0)
        f1 = f1_score(y_should_be_nei, y_predicted_nei, zero_division=0)
        accuracy = accuracy_score(y_should_be_nei, y_predicted_nei)

        # Confusion matrix: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(y_should_be_nei, y_predicted_nei).ravel()

        result = {
            "veracity_label": veracity_label,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
            "total_claims": len(veracity_df),
            "nei_predictions": int(y_predicted_nei.sum()),
            "nei_proportion": float(y_predicted_nei.mean()),
            "summary_in_rag_results": int((~veracity_df[rank_cutoff]).sum()),
            "summary_not_in_rag_results": int(veracity_df[rank_cutoff].sum()),
        }
        results.append(result)

    return results


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("Loading data...")
    retrieval_df = pd.read_parquet(RETRIEVAL_PATH)
    fact_checks_df = pd.read_parquet(
        FACTCHECKS_PATH,
        columns=["factcheck_analysis_link", "verdict"],
    )
    results_df = pd.read_parquet(
        RESULTS_PATH,
        columns=[
            "factcheck_analysis_link",
            "model",
            "label",
            "num_rag_items",
            "used_web_search",
            "prompt_tokens",
        ],
    )

    print("Merging data...")
    fact_checks_df = pd.merge(
        results_df,
        fact_checks_df,
        on="factcheck_analysis_link",
        how="left",
    )

    print("Creating retrieval indicators...")
    retrieval_df["in_top_3"] = retrieval_df["rank"].apply(
        lambda x: True if x <= 3 else False
    )
    retrieval_df["in_top_6"] = retrieval_df["rank"].apply(
        lambda x: True if x <= 6 else False
    )
    retrieval_df["in_top_9"] = retrieval_df["rank"].apply(
        lambda x: True if x <= 9 else False
    )

    print("Merging with retrieval data...")
    fcs_w_recall_hits = pd.merge(
        fact_checks_df,
        retrieval_df[["factcheck_analysis_link", "in_top_3", "in_top_6", "in_top_9"]],
        on="factcheck_analysis_link",
        how="left",
    )

    print("Analyzing NEI behavior by veracity...")
    performance_records = []
    confusion_records = []
    results_records = []

    total_groups = len(
        list(fcs_w_recall_hits.groupby(["model", "num_rag_items", "used_web_search"]))
    )
    processed_groups = 0

    for group, temp_df in fcs_w_recall_hits.groupby(
        ["model", "num_rag_items", "used_web_search"]
    ):
        processed_groups += 1
        if processed_groups % 10 == 0:
            print(f"Processing group {processed_groups}/{total_groups}")

        model_name = group[0]
        num_rag_items = group[1]
        used_web_search = group[2]

        # Skip non-RAG configurations
        if num_rag_items == 0:
            continue

        base_record = {
            "model": model_name,
            "num_rag_items": num_rag_items,
            "used_web_search": used_web_search,
        }

        # Calculate NEI behavior metrics by veracity label
        veracity_results = analyze_nei_behavior_by_veracity(
            temp_df, rank_cutoff=f"in_top_{int(num_rag_items)}"
        )

        # Process results for each veracity label
        for result in veracity_results:
            # Performance record
            performance_record = base_record.copy()
            performance_record.update(
                {
                    "veracity_label": result["veracity_label"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1_score": result["f1_score"],
                    "accuracy": result["accuracy"],
                }
            )
            performance_records.append(performance_record)

            # Confusion matrix record
            confusion_record = base_record.copy()
            confusion_record.update(
                {
                    "veracity_label": result["veracity_label"],
                    "true_positive": result["true_positive"],
                    "false_positive": result["false_positive"],
                    "true_negative": result["true_negative"],
                    "false_negative": result["false_negative"],
                }
            )
            confusion_records.append(confusion_record)

            # Results summary record
            results_record = base_record.copy()
            results_record.update(
                {
                    "veracity_label": result["veracity_label"],
                    "total_claims": result["total_claims"],
                    "nei_predictions": result["nei_predictions"],
                    "nei_proportion": result["nei_proportion"],
                    "summary_in_rag_results": result["summary_in_rag_results"],
                    "summary_not_in_rag_results": result["summary_not_in_rag_results"],
                }
            )
            results_records.append(results_record)

    print("Building output DataFrames...")
    # Build DataFrames from the records
    performance_df = (
        pd.DataFrame(performance_records)
        .sort_values(["model", "num_rag_items", "veracity_label"])
        .reset_index(drop=True)
    )

    confusion_df = (
        pd.DataFrame(confusion_records)
        .sort_values(["model", "num_rag_items", "veracity_label"])
        .reset_index(drop=True)
    )

    results_df = (
        pd.DataFrame(results_records)
        .sort_values(["model", "num_rag_items", "veracity_label"])
        .reset_index(drop=True)
    )

    print("Saving results...")
    # Save the results to parquet files
    performance_df.to_parquet(NEI_PERFORMANCE_METRICS_BY_VERACITY_PATH, index=False)
    confusion_df.to_parquet(NEI_CONFUSION_SUMMARY_BY_VERACITY_PATH, index=False)
    results_df.to_parquet(NEI_RESULTS_BY_VERACITY_PATH, index=False)

    print("Results saved to:")
    print(f"  - {NEI_PERFORMANCE_METRICS_BY_VERACITY_PATH}")
    print(f"  - {NEI_CONFUSION_SUMMARY_BY_VERACITY_PATH}")
    print(f"  - {NEI_RESULTS_BY_VERACITY_PATH}")

    print("\nSummary:")
    print(f"  - Performance metrics: {len(performance_df)} rows")
    print(f"  - Confusion matrices: {len(confusion_df)} rows")
    print(f"  - Results summary: {len(results_df)} rows")
    print(
        f"  - Veracity labels analyzed: {sorted(performance_df['veracity_label'].unique())}"
    )
    print(f"  - Models analyzed: {len(performance_df['model'].unique())}")
